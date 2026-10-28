from __future__ import annotations

import csv
import json
import os
import signal
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from src.checkpoint import (
    EvalRecord,
    RunJournal,
    StopRequested,
    eval_id,
    install_sigterm_handler,
    restore_sigterm_handler,
)
from src.config import Config
from src.evaluator import Evaluator
from src.misc import clean_run_outputs, output_prefixes
from src.searchMethods.tabu_flags import (
    _effective_signature,
    _render_flags,
    run_tabu_study,
)


def _fake_value(binary: str, env: dict) -> float:
    """
    Deterministic 'measurement'.

    Depends on the binary (which encodes the flags) and the environment, so the
    objective changes from configuration to configuration. ``crc32`` keeps the
    value stable across processes, which re-measuring would destroy.
    """
    value = 1.0 + (zlib.crc32(str(binary).encode()) % 1000) / 1000.0
    if env.get("MODE") == "b":
        value += 0.3
    return value


class TabuResumeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.calls = {"build": 0, "measure": 0}
        self._patches = [
            patch("src.evaluator.compile_single_source", self._fake_compile),
            patch("src.evaluator.measure_perf", self._fake_measure),
        ]
        for entry in self._patches:
            entry.start()
            self.addCleanup(entry.stop)
        self.addCleanup(restore_sigterm_handler)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _fake_compile(self, _compiler, _source, _flags, output, trial=None):
        self.calls["build"] += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("binary")
        output.chmod(0o755)
        return output

    def _fake_measure(self, _cfg, _binary, _args, env, _runs, **_kwargs):
        self.calls["measure"] += 1
        return {"CPI": _fake_value(str(_binary), env)}

    def write_config(self, **overrides) -> Path:
        source = self.root / "program.cpp"
        source.write_text("int main() { return 0; }\n")
        tabu = {
            "max_iters": 2,
            "max_no_improve": 20,
            "tabu_tenure": 4,
            "neighborhood": 3,
            "env_mode": "product",
            "env_cap": 2,
            "results_csv": str(self.root / "tabu.csv"),
        }
        tabu.update(overrides.pop("tabu", {}))
        raw = {
            "backend": "perf",
            "source": str(source),
            "compiler": "c++",
            "compiler_flags": ["-O2", "-O3"],
            "compiler_flag_pool": ["-ffast-math"],
            "compiler_params": {"-march": ["native"]},
            "compiler_params_select": {"min": 0, "max": 0},
            "program_args": ["-deffnm", "bench"],
            "env": {"MODE": ["a", "b"]},
            "objectives": [{"metric": "CPI", "goal": "min"}],
            "perf": {"events": ["cycles"]},
            "search": {"study": "tabu", "random_seed": 7},
            "tabu": tabu,
            "failed_builds": str(self.root / "tabu_failed.csv"),
        }
        raw.update(overrides)
        path = self.root / "config.json"
        path.write_text(json.dumps(raw))
        return path

    def load(self, **overrides) -> Config:
        return Config.load(self.write_config(**overrides))

    def run_study(self, cfg: Config, **kwargs) -> None:
        run_tabu_study(cfg, workroot=self.root / "work", **kwargs)

    def read_rows(self) -> list:
        with open(self.root / "tabu.csv", newline="") as fp:
            return list(csv.DictReader(fp))

    def read_state(self) -> dict:
        return json.loads((self.root / "tabu.state.json").read_text())

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_results_are_written_during_the_run(self):
        cfg = self.load()
        self.run_study(cfg, iters=2)
        rows = self.read_rows()
        self.assertTrue(rows)
        self.assertEqual(len(rows), self.calls["measure"])
        # Every measurement is on record with the columns needed to resume.
        for row in rows:
            self.assertTrue(row["eval_id"])
            self.assertTrue(row["flags"])
            self.assertTrue(row["compile_key"])
            self.assertEqual(row["CPI"], row["CPI"])  # objective column present
        self.assertEqual(self.read_state()["search"]["iteration"], 2)

    def test_resume_continues_and_reuses_recorded_results(self):
        # A larger pool keeps the search from exhausting the space, so four real
        # (non-equivalent) iterations are available.
        cfg = self.load(compiler_flag_pool=["-ffast-math", "-funroll-loops", "-fstrict-aliasing"])
        self.run_study(cfg, iters=2)
        rows_before = self.read_rows()

        self.run_study(cfg, resume=True, iters=4)
        rows_after = self.read_rows()

        # No configuration is measured (or recorded) twice.
        ids = [row["eval_id"] for row in rows_after]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(rows_after), self.calls["measure"])
        self.assertGreater(len(rows_after), len(rows_before))
        # Resume reuses earlier results instead of re-measuring them.
        self.assertEqual(rows_after[: len(rows_before)], rows_before)

        state = self.read_state()["search"]
        self.assertEqual(state["iteration"], 4)
        self.assertEqual(state["phase"], "search")
        self.assertIsNotNone(state["best"])
        self.assertTrue(state["best"]["key"])

    def test_resume_uses_existing_build_artifacts(self):
        cfg = self.load()
        self.run_study(cfg, iters=1)
        builds = self.calls["build"]
        self.assertGreater(builds, 0)

        # A fresh Evaluator on the same workroot must find the artifacts on disk
        # instead of rebuilding them.
        evaluator = Evaluator(cfg, self.root / "work")
        _, artifact = evaluator.get_or_build("-O2")
        self.assertTrue(artifact.is_file())
        self.assertEqual(self.calls["build"], builds)

    def test_missing_state_file_keeps_recorded_results(self):
        cfg = self.load(compiler_flag_pool=["-ffast-math", "-funroll-loops", "-fstrict-aliasing"])
        self.run_study(cfg, iters=2)
        rows_before = len(self.read_rows())
        (self.root / "tabu.state.json").unlink()

        self.run_study(cfg, resume=True, iters=4)

        rows = self.read_rows()
        ids = [row["eval_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(rows), self.calls["measure"])
        self.assertGreaterEqual(len(rows), rows_before)
        # The iteration count continues instead of restarting at zero.
        self.assertEqual(self.read_state()["search"]["iteration"], 4)

    def test_resume_refuses_changed_search_space(self):
        self.run_study(self.load(), iters=1)
        changed = self.load(compiler_flag_pool=["-funroll-loops"])
        with self.assertRaises(SystemExit):
            self.run_study(changed, resume=True, iters=2)

    def test_fresh_run_archives_previous_results(self):
        cfg = self.load()
        self.run_study(cfg, iters=1)
        self.run_study(cfg, iters=1)
        archives = list(self.root.glob("tabu.archive-*"))
        self.assertEqual(len(archives), 1)
        self.assertTrue((archives[0] / "tabu.csv").is_file())
        # The fresh run started from zero, so it did not inherit the old count.
        self.assertEqual(self.read_state()["search"]["iteration"], 1)

    def test_budget_modes(self):
        # A pool large enough that three real (non-equivalent) iterations are
        # available; the tiny default pool can be exhausted and then stall.
        cfg = self.load(compiler_flag_pool=["-ffast-math", "-funroll-loops", "-fstrict-aliasing"])
        self.run_study(cfg, iters=1, budget="lifetime")
        self.assertEqual(self.read_state()["search"]["iteration"], 1)

        self.run_study(cfg, resume=True, iters=2, budget="per-run")
        self.assertEqual(self.read_state()["search"]["iteration"], 3)

        self.run_study(cfg, resume=True, iters=2, budget="lifetime")
        self.assertEqual(self.read_state()["search"]["iteration"], 3)

    def test_sigterm_saves_state_and_stops_cleanly(self):
        cfg = self.load(tabu={"max_iters": 20, "max_no_improve": 20})
        limit = 3
        counter = {"n": 0}

        def dying_measure(_cfg, _binary, _args, env, _runs, **_kwargs):
            counter["n"] += 1
            self.calls["measure"] += 1
            if counter["n"] == limit:
                os.kill(os.getpid(), signal.SIGTERM)
                for _ in range(1000):  # give the handler a chance to run
                    pass
            return {"CPI": _fake_value(str(_binary), env)}

        with patch("src.evaluator.measure_perf", dying_measure):
            self.run_study(cfg, iters=20)

        rows = self.read_rows()
        self.assertEqual(len(rows), limit - 1)
        state = self.read_state()["search"]
        self.assertLess(state["iteration"], 20)
        self.assertTrue((self.root / "tabu.state.json").is_file())

    def test_flag_constraints_prevent_conflicting_pool_flags(self):
        cfg = self.load(
            compiler_flag_pool=["-funroll-loops", "-fno-unroll-loops"],
            flag_constraints={"exclusive": [["-funroll-loops", "-fno-unroll-loops"]]},
            tabu={
                "max_iters": 8, "max_no_improve": 20, "tabu_tenure": 4,
                "neighborhood": 6, "allow_variant_moves": False,
                "allow_param_moves": False, "allow_pool_moves": True,
                "allow_env_moves": False, "env_mode": "fixed", "env": {},
                "results_csv": str(self.root / "tabu.csv"),
            },
        )
        self.run_study(cfg, iters=8)
        rows = self.read_rows()
        self.assertTrue(rows)
        for row in rows:
            tokens = set((row["compiler_flags"] or "").split("|"))
            self.assertFalse(
                {"-funroll-loops", "-fno-unroll-loops"} <= tokens,
                row["compiler_flags"],
            )

    def test_flag_constraints_drop_flags_implied_by_variant(self):
        cfg = self.load(
            compiler_flags=["-O3"],
            compiler_flag_pool=["-funroll-loops", "-fno-unroll-loops"],
            flag_constraints={"implies": {"-O3": ["-funroll-loops"]}},
            tabu={
                "max_iters": 4, "max_no_improve": 20, "tabu_tenure": 4,
                "neighborhood": 6, "allow_variant_moves": True,
                "allow_param_moves": False, "allow_pool_moves": True,
                "allow_env_moves": False, "env_mode": "fixed", "env": {},
                "results_csv": str(self.root / "tabu.csv"),
            },
        )
        self.run_study(cfg, iters=4)
        rows = self.read_rows()
        self.assertTrue(rows)
        for row in rows:
            tokens = set((row["compiler_flags"] or "").split("|"))
            self.assertIn("-O3", tokens)
            self.assertNotIn("-funroll-loops", tokens)

    def test_equivalent_node_reuses_result_without_counting(self):
        cfg = self.load(
            compiler_flags_base="-O3",
            compiler_flags=[],
            compiler_flag_pool=["-funroll-loops"],
            flag_constraints={"implies": {"-O3": ["-funroll-loops"]}},
            tabu={
                "max_iters": 10, "max_no_improve": 20, "max_skips": 3,
                "tabu_tenure": 1, "neighborhood": 4,
                "allow_variant_moves": False, "allow_param_moves": False,
                "allow_pool_moves": True, "allow_env_moves": False,
                "env_mode": "fixed", "env": {},
                "results_csv": str(self.root / "tabu.csv"),
            },
        )
        self.run_study(cfg, iters=10)
        # Only the baseline effective configuration (-O3) is ever measured:
        # "-O3 -funroll-loops" is equivalent because -O3 implies -funroll-loops.
        self.assertEqual(len(self.read_rows()), 1)
        self.assertEqual(self.calls["measure"], 1)
        # Equivalent nodes are still traversed but do not count as iterations.
        self.assertEqual(self.read_state()["search"]["iteration"], 0)

    def test_multitoken_pool_entries_are_not_tokenised(self):
        cfg = self.load(
            compiler_flag_pool=[
                "-mllvm -force-vector-interleave={2,4,8,16}",
                "-mllvm -enable-loopinterchange",
            ],
            flag_constraints={"exclusive": [["-funroll-loops", "-fno-unroll-loops"]]},
        )
        entries = [
            "-O2",
            "-mllvm -enable-loop-flatten",
            "-mllvm -enable-unroll-and-jam",
        ]
        # Entries are opaque: a pool item that contains a space must survive
        # intact, including its repeated '-mllvm'.
        self.assertEqual(list(_effective_signature(cfg, entries)), entries)

        key, flags_str, parts = _render_flags(cfg, "", "-O2", {}, entries[1:])
        self.assertEqual(parts, entries)
        self.assertEqual(flags_str.count("-mllvm"), 2)
        self.assertIn("-mllvm -enable-loop-flatten", key)
        self.assertIn("-mllvm -enable-unroll-and-jam", key)

    def test_multitoken_entries_survive_into_results_csv(self):
        cfg = self.load(
            compiler_flag_pool=[
                "-mllvm -enable-loop-flatten",
                "-mllvm -enable-unroll-and-jam",
            ],
            flag_constraints={"exclusive": [["-funroll-loops", "-fno-unroll-loops"]]},
            tabu={
                "max_iters": 4, "max_no_improve": 20, "tabu_tenure": 4,
                "neighborhood": 6, "allow_variant_moves": False,
                "allow_param_moves": False, "allow_pool_moves": True,
                "allow_env_moves": False, "env_mode": "fixed", "env": {},
                "results_csv": str(self.root / "tabu.csv"),
            },
        )
        self.run_study(cfg, iters=4)
        rows = self.read_rows()
        self.assertTrue(rows)
        for row in rows:
            key = row["compiler_flags"] or ""
            # Every generated configuration keeps the full multi-token entry.
            if "-enable-loop-flatten" in key:
                self.assertIn("-mllvm -enable-loop-flatten", key)
            if "-enable-unroll-and-jam" in key:
                self.assertIn("-mllvm -enable-unroll-and-jam", key)

    def test_resume_refuses_changed_flag_constraints(self):
        self.run_study(
            self.load(flag_constraints={"exclusive": [["-funroll-loops", "-fno-unroll-loops"]]}),
            iters=1,
        )
        changed = self.load(flag_constraints={"exclusive": [["-ffast-math", "-fno-fast-math"]]})
        with self.assertRaises(SystemExit):
            self.run_study(changed, resume=True, iters=2)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_study(self.load(), iters=1, budget="forever")

    def test_journal_round_trip(self):
        journal = RunJournal(
            self.root / "j.csv",
            objective_metrics=["CPI"],
            failure_path=self.root / "j_failed.csv",
        )
        journal.append(
            EvalRecord(
                k=0,
                eval_id="abc",
                value=1.5,
                flags_key="-O2|MODE=a",
                flags="-O2",
                env={"MODE": "a"},
                binary="/tmp/bin",
                compile_key="deadbeef",
                metrics={"CPI": 1.5, "cycles": 100.0},
            )
        )
        # A later row introduces a new metric column without corrupting the first.
        journal.append(
            EvalRecord(
                k=1,
                eval_id="def",
                value=2.0,
                flags_key="-O3|MODE=b",
                flags="-O3",
                env={"MODE": "b"},
                binary="/tmp/bin2",
                compile_key="cafebabe",
                metrics={"CPI": 2.0, "instructions": 42.0},
            )
        )
        journal.record_failure(k=1, eid="bad", flags="-O0", env={"MODE": "a"}, reason="boom")

        reopened = RunJournal(
            self.root / "j.csv",
            objective_metrics=["CPI"],
            failure_path=self.root / "j_failed.csv",
        )
        self.assertIsNone(reopened.load())

        record = reopened.lookup("abc")
        self.assertIsNotNone(record)
        self.assertEqual(record.flags, "-O2")
        self.assertEqual(record.env, {"MODE": "a"})
        self.assertAlmostEqual(record.metrics["cycles"], 100.0)
        self.assertAlmostEqual(reopened.lookup("def").value, 2.0)
        self.assertEqual(reopened.failure_reason("bad"), "boom")
        self.assertEqual(reopened.known_failures()["bad"]["flags"], "-O0")
        self.assertIsNotNone(reopened.lookup("abc"))
        # Both metric columns survived the header rewrite.
        with open(self.root / "j.csv", newline="") as fp:
            header = next(csv.reader(fp))
        self.assertIn("cycles", header)
        self.assertIn("instructions", header)

    def test_objective_stdev_is_recorded_in_the_results_csv(self):
        def measure_with_stdev(_cfg, _binary, _args, env, _runs, **_kwargs):
            return {"CPI": _fake_value(str(_binary), env), "CPI_stdev": 0.125}

        with patch("src.evaluator.measure_perf", measure_with_stdev):
            self.run_study(self.load(), iters=2)

        rows = self.read_rows()
        self.assertTrue(rows)
        self.assertIn("CPI_stdev", rows[0])
        for row in rows:
            self.assertAlmostEqual(float(row["CPI_stdev"]), 0.125)
        # The stddev is an extra column next to the objective, not a replacement.
        with open(self.root / "tabu.csv", newline="") as fp:
            header = next(csv.reader(fp))
        self.assertIn("CPI", header)
        self.assertIn("CPI_stdev", header)

    def test_objective_stdev_survives_resume(self):
        def measure_with_stdev(_cfg, _binary, _args, env, _runs, **_kwargs):
            return {"CPI": _fake_value(str(_binary), env), "CPI_stdev": 0.5}

        with patch("src.evaluator.measure_perf", measure_with_stdev):
            cfg = self.load(compiler_flag_pool=["-ffast-math", "-funroll-loops", "-fstrict-aliasing"])
            self.run_study(cfg, iters=2)
            self.run_study(cfg, resume=True, iters=4)

        rows = self.read_rows()
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("CPI_stdev", row)

    def test_journal_round_trips_metric_stdev(self):
        journal = RunJournal(
            self.root / "s.csv",
            objective_metrics=["CPI"],
            failure_path=self.root / "s_failed.csv",
        )
        journal.append(
            EvalRecord(
                k=0, eval_id="e1", value=1.0, flags_key="-O2", flags="-O2",
                env={"MODE": "a"}, binary="/tmp/bin", compile_key="c0",
                metrics={"CPI": 1.0, "CPI_stdev": 0.01},
            )
        )
        reopened = RunJournal(
            self.root / "s.csv",
            objective_metrics=["CPI"],
            failure_path=self.root / "s_failed.csv",
        )
        reopened.load()
        with open(self.root / "s.csv", newline="") as fp:
            header = next(csv.reader(fp))
        self.assertIn("CPI_stdev", header)
        self.assertAlmostEqual(reopened.lookup("e1").metrics["CPI_stdev"], 0.01)

    def test_eval_id_identifies_flags_and_environment(self):
        self.assertEqual(eval_id("k", {"A": "1", "B": "2"}), eval_id("k", {"B": "2", "A": "1"}))
        self.assertNotEqual(eval_id("k", {"A": "1"}), eval_id("k", {"A": "2"}))
        self.assertNotEqual(eval_id("k1", {"A": "1"}), eval_id("k2", {"A": "1"}))

    def test_stop_requested_is_not_swallowed_by_generic_handlers(self):
        try:
            raise StopRequested("stop")
        except Exception:  # pragma: no cover - must not be reached
            self.fail("StopRequested must not be an Exception subclass")
        except StopRequested:
            pass

    def test_sigterm_handler_raises_stop_requested(self):
        install_sigterm_handler()
        with self.assertRaises(StopRequested):
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(1000):  # pragma: no cover - handler runs immediately
                pass
        self.assertEqual(1, 1)

    def test_output_prefixes_and_cleanup(self):
        self.assertEqual(
            output_prefixes(["-deffnm", "bench", "-nsteps", "10"], ["*.tmp"]),
            ["*.tmp", "bench.*"],
        )
        run_dir = self.root / "run"
        run_dir.mkdir()
        for name in ("bench.log", "bench.edr", "keep.txt"):
            (run_dir / name).write_text("x")
        removed = clean_run_outputs(run_dir, ["-deffnm", "bench"])
        self.assertEqual(sorted(p.name for p in removed), ["bench.edr", "bench.log"])
        self.assertTrue((run_dir / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()
