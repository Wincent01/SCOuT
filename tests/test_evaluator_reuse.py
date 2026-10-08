from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import build
from src.config import BuildProject, Config
from src.evaluator import Evaluator


class EvaluatorReuseTests(unittest.TestCase):
    """
    Regression tests for build-artifact reuse.

    Only artifacts that live inside the per-key build directory may be cached or
    persisted. Raw build outputs elsewhere (a "gmx" binary built by make, or an
    absolute project.executable) are shared between configurations, so a later
    build overwrites them and they must never be served from the build cache.
    """

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    # ------------------------------------------------------------------
    # config helpers
    # ------------------------------------------------------------------
    def _source_config(self) -> Path:
        source = self.root / "program.cpp"
        source.write_text("int main() { return 0; }\n")
        raw = {
            "backend": "perf",
            "source": str(source),
            "compiler": "c++",
            "compiler_flags": ["-O2"],
            "objectives": [{"metric": "CPI", "goal": "min"}],
            "perf": {"events": ["cycles"]},
            "search": {"study": "optuna", "sampler": "rs"},
        }
        path = self.root / "source_config.json"
        path.write_text(json.dumps(raw))
        return path

    def _project_config(self) -> Path:
        project_dir = self.root / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "backend": "perf",
            "project": {
                "dir": str(project_dir),
                "build_system": "make",
                "target": "gmx",
                "executable": "bin/gmx",
                "make_flags_var": "EXTRA_CFLAGS",
            },
            "compiler": "c++",
            "compiler_flags": ["-O2"],
            "objectives": [{"metric": "CPI", "goal": "min"}],
            "perf": {"events": ["cycles"]},
            "search": {"study": "optuna", "sampler": "rs"},
        }
        path = self.root / "project_config.json"
        path.write_text(json.dumps(raw))
        return path

    # ------------------------------------------------------------------
    # evaluator cacheability
    # ------------------------------------------------------------------
    def test_shared_output_binary_is_not_cached(self):
        cfg = Config.load(self._project_config())
        shared = self.root / "project" / "bin" / "gmx"
        seen: list[str] = []

        def fake_compile(_project, _compiler, flags, _build_dir, trial=None):
            # A build system that always writes to one shared location, like a
            # make project whose output is copied by the evaluator is not.
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_text(f"binary:{flags}")
            shared.chmod(0o755)
            seen.append(flags)
            return shared

        with patch("src.evaluator.compile_project", fake_compile):
            evaluator = Evaluator(cfg, self.root / "work")
            key_a, bin_a = evaluator.get_or_build("-O2")
            key_b, bin_b = evaluator.get_or_build("-O3")

            self.assertNotEqual(key_a, key_b)
            # Both configurations resolve to the same shared path; the second
            # build has overwritten the first configuration's binary.
            self.assertEqual(bin_a, bin_b)
            self.assertEqual(bin_b.read_text(), "binary:-O3")
            self.assertEqual(evaluator.build_count, 0)

            # Re-requesting the first configuration must rebuild it rather than
            # returning the shared path that now holds the other binary.
            key_a2, bin_a2 = evaluator.get_or_build("-O2")

        self.assertEqual(key_a2, key_a)
        self.assertEqual(bin_a2.read_text(), "binary:-O2")
        self.assertEqual(len(seen), 3)

    def test_copied_artifact_inside_build_dir_is_cached(self):
        cfg = Config.load(self._source_config())
        calls = {"build": 0}

        def fake_compile(_compiler, _source, _flags, output, trial=None):
            calls["build"] += 1
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("binary")
            output.chmod(0o755)
            return output

        with patch("src.evaluator.compile_single_source", fake_compile):
            evaluator = Evaluator(cfg, self.root / "work")
            _, first = evaluator.get_or_build("-O2")
            _, second = evaluator.get_or_build("-O2")

        self.assertEqual(first, second)
        self.assertEqual(calls["build"], 1)
        self.assertEqual(evaluator.build_count, 1)

    def test_gmx_output_inside_build_dir_is_reused_within_run(self):
        cfg = Config.load(self._project_config())
        calls: list[str] = []

        def fake_compile(_project, _compiler, flags, build_dir, trial=None):
            output = Path(build_dir) / "cmake_x" / "bin" / "gmx"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"binary:{flags}")
            output.chmod(0o755)
            calls.append(flags)
            return output

        with patch("src.evaluator.compile_project", fake_compile):
            evaluator = Evaluator(cfg, self.root / "work")
            _, first = evaluator.get_or_build("-O2")
            _, second = evaluator.get_or_build("-O2")
            _, other = evaluator.get_or_build("-O3")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(calls), 2)
        self.assertEqual(evaluator.build_count, 2)

    # ------------------------------------------------------------------
    # make freshness
    # ------------------------------------------------------------------
    def test_make_build_removes_declared_target_and_forces_rebuild(self):
        commands: list = []

        def fake_run(cmd, **_kwargs):
            commands.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        declared = self.root / "project" / "gmx"
        declared.parent.mkdir(parents=True, exist_ok=True)
        declared.write_text("stale binary from a previous run")
        declared.chmod(0o755)

        fresh = self.root / "fresh-binary"
        fresh.write_text("fresh")
        fresh.chmod(0o755)

        project = BuildProject(
            self.root / "project", "make", target="gmx", make_flags_var="EXTRA_CFLAGS"
        )
        with patch.object(build, "_run", fake_run), patch.object(
            build, "_resolve_project_executable", lambda *_args: fresh
        ):
            result = build.compile_project(project, "c++", "-O3", self.root / "work")

        self.assertEqual(result, fresh)
        # The stale, explicitly declared target is removed before building.
        self.assertFalse(declared.exists())
        # And the build ignores timestamps so it cannot reuse an up-to-date target.
        make_builds = [
            cmd for cmd in commands
            if isinstance(cmd, list) and cmd[:1] == ["make"] and "--always-make" in cmd
        ]
        self.assertTrue(make_builds)
        self.assertIn("EXTRA_CFLAGS+=-O3", make_builds[0])


if __name__ == "__main__":
    unittest.main()
