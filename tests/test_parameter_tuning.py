from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from unittest.mock import patch

from src import build
from src.config import BuildProject, Config, LikwidConfig, MetricSpec, ParserConfig, PerfConfig
from src.evaluator import Evaluator
from src.metrics import (
    _SYCL_RE,
    _measurement_env,
    measure_likwid,
    measure_parser_sycl,
    measure_perf,
    resolve_run_cwd,
)
from src.misc import _render_one_param
from src.searchMethods.anneal_flags import _score as anneal_score


class ParameterTuningTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_config(self, **overrides) -> Path:
        source = self.root / "program.cpp"
        source.write_text("int main() { return 0; }\n")
        raw = {
            "backend": "perf",
            "source": str(source),
            "compiler": "c++",
            "compiler_flags": ["-O2"],
            "env": {"MODE": ["a", "b"]},
            "objectives": [{"metric": "CPI", "goal": "min"}],
            "perf": {"events": ["cycles", "instructions"]},
            "search": {"study": "optuna", "sampler": "rs"},
        }
        raw.update(overrides)
        path = self.root / "config.json"
        path.write_text(json.dumps(raw))
        return path

    def test_evaluator_builds_once_for_many_environments(self):
        cfg = Config.load(self.write_config())
        calls = {"build": 0, "measure": 0}

        def fake_compile(_compiler, _source, _flags, output, trial=None):
            calls["build"] += 1
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("binary")
            output.chmod(0o755)
            return output

        def fake_measure(_cfg, _binary, _args, env, _runs, **_kwargs):
            calls["measure"] += 1
            return {"CPI": 1.0 if env["MODE"] == "a" else 2.0}

        with patch("src.evaluator.compile_single_source", fake_compile), patch(
            "src.evaluator.measure_perf", fake_measure
        ):
            evaluator = Evaluator(cfg, self.root / "work")
            evaluator.evaluate("-O2", {"MODE": "a"}, self.root / "run-a")
            evaluator.evaluate("-O2", {"MODE": "b"}, self.root / "run-b")
            cached = evaluator.evaluate("-O2", {"MODE": "a"}, self.root / "run-a2")

        self.assertEqual(calls, {"build": 1, "measure": 2})
        self.assertTrue(cached.cached)
        self.assertEqual(evaluator.build_count, 1)

    def test_config_preserves_custom_search_settings(self):
        cfg = Config.load(self.write_config(
            search={"study": "beam_tabu", "random_seed": 7},
            beam_tabu={"beam_width": 3, "env_cap": 1},
            significance={"min_rel_gain": 0.05},
        ))
        self.assertEqual(cfg.beam_tabu["beam_width"], 3)
        self.assertEqual(cfg.beam_tabu["env_cap"], 1)
        self.assertAlmostEqual(cfg.significance.min_rel_gain, 0.05)

    def test_cmake_uses_real_flag_variable(self):
        commands = []

        def fake_run(cmd, **_kwargs):
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        fake_binary = self.root / "program"
        fake_binary.write_text("binary")
        fake_binary.chmod(0o755)
        project = BuildProject(self.root, "cmake", target="program")
        with patch.object(build, "_run", fake_run), patch.object(
            build, "_resolve_project_executable", lambda *_args: fake_binary
        ):
            result = build.compile_project(project, "c++", "-O3", self.root / "work")
        self.assertEqual(result, fake_binary)
        self.assertIn("-DCMAKE_CXX_FLAGS:STRING=-O3", commands[0])
        self.assertFalse(any("CMAKE_CXX_FLAGS+=" in arg for arg in commands[0]))

    def test_project_target_and_executable_are_separate(self):
        executable = self.root / "bin" / "program"
        executable.parent.mkdir()
        executable.write_text("binary")
        executable.chmod(0o755)
        project = BuildProject(self.root, "make", target="all", executable=Path("bin/program"))
        self.assertEqual(build._resolve_project_executable(project, self.root), executable)

    def test_managed_environment_keys_do_not_leak(self):
        previous = os.environ.get("CONDITIONAL_SETTING")
        os.environ["CONDITIONAL_SETTING"] = "ambient"
        try:
            merged = _measurement_env({"MODE": "generic"}, ["MODE", "CONDITIONAL_SETTING"])
        finally:
            if previous is None:
                os.environ.pop("CONDITIONAL_SETTING", None)
            else:
                os.environ["CONDITIONAL_SETTING"] = previous
        self.assertEqual(merged["MODE"], "generic")
        self.assertNotIn("CONDITIONAL_SETTING", merged)

    def test_compiler_parameter_template_is_rendered(self):
        trial = SimpleNamespace(suggest_categorical=lambda _name, values: values[0])
        key, flag = _render_one_param(trial, "--tile={}", [16])
        self.assertEqual(key, "--tile=16")
        self.assertEqual(flag, "--tile=16")

    def test_run_cwd_resolver_options(self):
        binary = self.root / "build" / "bin" / "program"
        binary.parent.mkdir(parents=True)
        binary.write_text("x")
        project = BuildProject(self.root, "cmake", target="program")

        # default / process -> inherit SCOuT's cwd
        self.assertIsNone(resolve_run_cwd("process", binary, self.root / "wd", project))
        self.assertIsNone(resolve_run_cwd(None, binary, self.root / "wd", project))
        self.assertEqual(resolve_run_cwd("binary_dir", binary, self.root / "wd", project), binary.parent)
        self.assertEqual(resolve_run_cwd("project_dir", binary, self.root / "wd", project), self.root)
        self.assertEqual(resolve_run_cwd("workdir", binary, self.root / "wd", project), self.root / "wd")
        # unknown -> binary_dir, matching the parser fallback
        self.assertEqual(resolve_run_cwd("whatever", binary, None, None), binary.parent)

    def test_measure_perf_runs_in_configured_cwd(self):
        binary = self.root / "build" / "bin" / "program"
        binary.parent.mkdir(parents=True)
        binary.write_text("x")
        seen = []

        def fake_run(cmd, *, cwd=None, env=None):
            seen.append(cwd)
            return SimpleNamespace(returncode=0, stdout="", stderr="1000 cycles\n500 instructions\n")

        with patch("src.metrics._run", fake_run):
            measure_perf(
                PerfConfig(events=["cycles", "instructions"], core_list=None, run_cwd="binary_dir"),
                binary, ["-nsteps", "1"], {}, 1,
                clear_runtime_cache=False,
            )
        self.assertEqual(seen, [binary.parent])

        seen.clear()
        with patch("src.metrics._run", fake_run):
            measure_perf(
                PerfConfig(events=["cycles"], core_list=None),
                binary, [], {}, 1,
                clear_runtime_cache=False,
            )
        self.assertEqual(seen, [None])  # None -> child inherits SCOuT's cwd

    def test_measure_likwid_runs_in_configured_cwd(self):
        binary = self.root / "build" / "bin" / "program"
        binary.parent.mkdir(parents=True)
        binary.write_text("x")
        seen = []

        def fake_run(cmd, *, cwd=None, env=None):
            seen.append(cwd)
            return SimpleNamespace(returncode=0, stdout="| cycles | 100 |\n", stderr="")

        likwid = LikwidConfig(
            group=None, events=["cycles"], metrics=[MetricSpec(name="cycles")],
            core_list=None, run_cwd="workdir",
        )
        with patch("src.metrics._run", fake_run):
            measure_likwid(
                likwid, binary, [], {}, 1,
                workdir=self.root / "wd", clear_runtime_cache=False,
            )
        self.assertEqual(seen, [self.root / "wd"])

    def test_run_cwd_config_is_validated(self):
        cfg = Config.load(self.write_config(perf={"events": ["cycles"], "run_cwd": "binary_dir"}))
        self.assertEqual(cfg.perf.run_cwd, "binary_dir")
        self.assertEqual(
            Config.load(self.write_config(perf={"events": ["cycles"]})).perf.run_cwd,
            "process",
        )
        with self.assertRaisesRegex(ValueError, "run_cwd"):
            Config.load(self.write_config(perf={"events": ["cycles"], "run_cwd": "nonsense"}))

    def test_parser_accepts_scientific_notation(self):
        match = _SYCL_RE.search("[SYCL][avg] kernel 2: 1.25e-06 s over 10 iters")
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group("val")), 1.25e-6)

    def test_maximization_score_orders_finite_values_correctly(self):
        self.assertLess(anneal_score(2.0, "max"), anneal_score(1.0, "max"))

    def test_invalid_search_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "search.study"):
            Config.load(self.write_config(search={"study": "typo"}))

    def test_repository_cmake_example_loads(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = Config.load(repo_root / "configs/example_configs/cmake_project.json")
        self.assertIsNotNone(cfg.project)
        self.assertEqual(cfg.project.build_system, "cmake")
        self.assertEqual(cfg.project.target, "bude")
        self.assertEqual(cfg.project.executable, Path("bude"))
        self.assertEqual(cfg.project.cmake_flag_vars, ["CXX_EXTRA_FLAGS"])

    def test_core_cli_does_not_import_polymorph(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = """
import builtins
original_import = builtins.__import__
def import_without_polymorph(name, *args, **kwargs):
    if name.startswith('src.polyMorph'):
        raise AssertionError(f'core CLI imported optional module: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_polymorph
import main
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_polymorph_package_keeps_runner_lazy(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = """
import sys
import src.polyMorph
assert 'src.polyMorph.runner' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    # ------------------------------------------------------------------
    # Standard deviation across measurement runs (sample stdev, <metric>_stdev)
    # ------------------------------------------------------------------
    def test_measure_perf_reports_run_stdev(self):
        binary = self.root / "program"
        binary.write_text("x")
        counts = [1000.0, 1100.0, 1300.0]
        calls = {"i": 0}

        def fake_run(cmd, *, cwd=None, env=None):
            val = counts[calls["i"] % len(counts)]
            calls["i"] += 1
            return SimpleNamespace(returncode=0, stdout="", stderr=f"{int(val)} cycles\n500 instructions\n")

        with patch("src.metrics._run", fake_run):
            mets = measure_perf(
                PerfConfig(events=["cycles", "instructions"], core_list=None),
                binary, [], {}, 3, clear_runtime_cache=False,
            )

        self.assertAlmostEqual(mets["cycles"], mean(counts))
        self.assertAlmostEqual(mets["cycles_stdev"], stdev(counts))
        # Constant metric still reports a (zero) stddev once there are 2+ samples.
        self.assertEqual(mets["instructions_stdev"], 0.0)

    def test_measure_perf_single_run_omits_stdev(self):
        binary = self.root / "program"
        binary.write_text("x")

        def fake_run(cmd, *, cwd=None, env=None):
            return SimpleNamespace(returncode=0, stdout="", stderr="1000 cycles\n")

        with patch("src.metrics._run", fake_run):
            mets = measure_perf(
                PerfConfig(events=["cycles"], core_list=None),
                binary, [], {}, 1, clear_runtime_cache=False,
            )

        self.assertIn("cycles", mets)
        self.assertNotIn("cycles_stdev", mets)

    def test_measure_likwid_reports_run_stdev(self):
        binary = self.root / "program"
        binary.write_text("x")
        values = [10.0, 20.0, 40.0]
        calls = {"i": 0}

        def fake_run(cmd, *, cwd=None, env=None):
            val = values[calls["i"] % len(values)]
            calls["i"] += 1
            return SimpleNamespace(returncode=0, stdout=f"| cycles | {val} |\n", stderr="")

        likwid = LikwidConfig(
            group=None, events=["cycles"], metrics=[MetricSpec(name="cycles")], core_list=None,
        )
        with patch("src.metrics._run", fake_run):
            mets = measure_likwid(likwid, binary, [], {}, 3, clear_runtime_cache=False)

        self.assertAlmostEqual(mets["cycles"], mean(values))
        self.assertAlmostEqual(mets["cycles_stdev"], stdev(values))

    def test_measure_parser_gromacs_reports_run_stdev(self):
        binary = self.root / "program"
        binary.write_text("x")
        values = [100.0, 120.0, 140.0]
        calls = {"i": 0}

        def fake_run(cmd, *, cwd=None, env=None):
            val = values[calls["i"] % len(values)]
            calls["i"] += 1
            return SimpleNamespace(returncode=0, stdout=f"Performance: {val} 0.1 0.2 0.3\n", stderr="")

        cfg = ParserConfig(format="gromacs", run_cwd="binary_dir")
        with patch("src.metrics._run", fake_run):
            mets = measure_parser_sycl(cfg, binary, [], {}, 3, clear_runtime_cache=False)

        self.assertAlmostEqual(mets["gmx_ns_per_day"], mean(values))
        self.assertAlmostEqual(mets["gmx_ns_per_day_stdev"], stdev(values))

    def _write_gmx_device_log(self, mapping):
        (self.root / "scout.log").write_text(
            "  GPU info:\n"
            "    Number of GPUs detected: 2\n"
            "    #0: name: AdaptiveCpp OpenMP host device, vendor: the AdaptiveCpp project, "
            "device version: AdaptiveCpp 25.10, driver version: 1.2, status: compatible\n"
            "    #1: name: cpu-skylake-avx512-AMD EPYC 9354 32-Core Processor, vendor: AuthenticAMD, "
            "device version: OpenCL 3.0 PoCL HSTR: cpu-x86_64, driver version: 6.0, status: compatible\n"
            "\n"
            "1 GPU selected for this run.\n"
            "Mapping of GPU IDs to the 1 GPU task in the 1 rank on this node:\n"
            f"  {mapping}\n"
            "PP tasks will do (non-perturbed) short-ranged interactions on the GPU\n"
        )

    def test_gromacs_device_report_checks_selected_device(self):
        binary = self.root / "program"
        binary.write_text("x")

        def fake_run(cmd, *, cwd=None, env=None):
            return SimpleNamespace(returncode=0, stdout="Performance: 1.0 0.1 0.2 0.3\n", stderr="acpp debug")

        args = ["mdrun", "-deffnm", "scout"]
        cfg = ParserConfig(format="gromacs", run_cwd="binary_dir", expect_device="PoCL")
        self._write_gmx_device_log("PP:1")
        with patch("src.metrics._run", fake_run):
            mets = measure_parser_sycl(cfg, binary, args, {}, 1, workdir=self.root, clear_runtime_cache=False)
        self.assertAlmostEqual(mets["gmx_ns_per_day"], 1.0)
        self.assertEqual((self.root / "parser_logs" / "device_run.err").read_text(), "acpp debug")

        self._write_gmx_device_log("PP:0")
        with patch("src.metrics._run", fake_run), self.assertRaisesRegex(RuntimeError, "OpenMP host device"):
            measure_parser_sycl(cfg, binary, args, {}, 1, workdir=self.root, clear_runtime_cache=False)

        (self.root / "scout.log").write_text("  GPU info:\n    Number of GPUs detected: 0\n\n")
        with patch("src.metrics._run", fake_run), self.assertRaisesRegex(RuntimeError, "no GPU tasks"):
            measure_parser_sycl(cfg, binary, args, {}, 1, workdir=self.root, clear_runtime_cache=False)

    def test_measure_parser_sycl_reports_run_stdev(self):
        binary = self.root / "program"
        binary.write_text("x")
        values = [0.001, 0.002, 0.004]
        calls = {"i": 0}

        def fake_run(cmd, *, cwd=None, env=None):
            val = values[calls["i"] % len(values)]
            calls["i"] += 1
            return SimpleNamespace(
                returncode=0, stdout=f"[SYCL][avg] kernel 1: {val} s over 10 iters\n", stderr=""
            )

        cfg = ParserConfig(format="sycl", aggregate="sum", run_cwd="binary_dir")
        with patch("src.metrics._run", fake_run):
            mets = measure_parser_sycl(cfg, binary, [], {}, 3, clear_runtime_cache=False)

        self.assertAlmostEqual(mets["sycl_kernel_1_avg_s"], mean(values))
        self.assertAlmostEqual(mets["sycl_kernel_1_avg_s_stdev"], stdev(values))
        self.assertAlmostEqual(mets["sycl_avg_sum_s"], mean(values))
        self.assertAlmostEqual(mets["sycl_avg_sum_s_stdev"], stdev(values))
        self.assertNotIn("sycl_iters_stdev", mets)


if __name__ == "__main__":
    unittest.main()
