from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import Config
from src.flag_constraints import FlagConstraints
from src.misc import suggest_compiler_flags


def _sample_constraints() -> FlagConstraints:
    return FlagConstraints.from_dict({
        "exclusive": [
            ["-funroll-loops", "-fno-unroll-loops"],
            ["-fstrict-aliasing", "-fno-strict-aliasing"],
        ],
        "implies": {
            "-O2": ["-fstrict-aliasing"],
            "-O3": ["-fstrict-aliasing", "-funroll-loops"],
            "-ffast-math": ["-fno-math-errno", "-ffp-contract=fast"],
        },
    })


class StubTrial:
    """Minimal Optuna-trial double driven by a lookup table."""

    def __init__(self, table=None):
        self.table = dict(table or {})
        self.number = 100  # stage 2 of _complexity_limits -> full caps

    def suggest_categorical(self, name, values):
        return self.table.get(name, values[0])

    def suggest_int(self, name, low, high):
        return self.table.get(name, high)

    def suggest_float(self, name, low, high):
        return self.table.get(name, 1.0)


class FlagConstraintsUnitTests(unittest.TestCase):
    def test_empty_constraints_are_inert(self):
        c = FlagConstraints.from_dict(None)
        self.assertTrue(c.is_empty)
        self.assertTrue(c.can_add("-ffast-math", ["-O3"]))
        self.assertTrue(c.is_allowed(["-funroll-loops", "-fno-unroll-loops"]))
        self.assertEqual(c.sanitize(["-a", "-b"]), ["-a", "-b"])
        self.assertEqual(c.select_consistent(["-a", "-b"]), ["-a", "-b"])

    def test_conflicts_and_can_add(self):
        c = _sample_constraints()
        self.assertTrue(c.conflicts("-funroll-loops", ["-fno-unroll-loops"]))
        self.assertFalse(c.conflicts("-funroll-loops", ["-O2"]))
        self.assertTrue(c.can_add("-funroll-loops", ["-O2"]))
        self.assertFalse(c.can_add("-funroll-loops", ["-fno-unroll-loops"]))
        self.assertFalse(c.can_add("-funroll-loops", ["-O3"]))

    def test_implies_is_transitive(self):
        c = FlagConstraints.from_dict({
            "implies": {
                "-O3": ["-ffast-math"],
                "-ffast-math": ["-fno-math-errno"],
            }
        })
        self.assertEqual(
            c.implied_flags(["-O3"]),
            frozenset({"-ffast-math", "-fno-math-errno"}),
        )

    def test_prune_pool_removes_implied(self):
        c = _sample_constraints()
        pool = ["-funroll-loops", "-fno-unroll-loops", "-fno-math-errno"]
        self.assertEqual(
            c.prune_pool(pool, ["-O3"]),
            ["-fno-unroll-loops", "-fno-math-errno"],
        )
        self.assertEqual(
            c.prune_pool(pool, ["-ffast-math"]),
            ["-funroll-loops", "-fno-unroll-loops"],
        )

    def test_select_consistent_drops_redundant_order_independently(self):
        c = _sample_constraints()
        self.assertEqual(
            c.select_consistent(["-ffp-contract=fast", "-ffast-math"], []),
            ["-ffast-math"],
        )
        self.assertEqual(
            c.select_consistent(["-ffast-math", "-ffp-contract=fast"], []),
            ["-ffast-math"],
        )

    def test_select_consistent_keeps_one_of_exclusive_pair(self):
        c = _sample_constraints()
        self.assertEqual(
            c.select_consistent(["-funroll-loops", "-fno-unroll-loops"], []),
            ["-funroll-loops"],
        )

    def test_sanitize_respects_order_and_exclusivity(self):
        c = _sample_constraints()
        self.assertEqual(c.sanitize(["-O3", "-funroll-loops"]), ["-O3"])
        self.assertTrue(c.is_allowed(["-O3", "-fno-unroll-loops"]))
        self.assertFalse(c.is_allowed(["-O3", "-funroll-loops"]))
        self.assertFalse(c.is_consistent(["-funroll-loops", "-fno-unroll-loops"]))

    def test_malformed_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            FlagConstraints.from_dict({"exclusive": [["only-one"]]})
        with self.assertRaises(ValueError):
            FlagConstraints.from_dict({"bogus": []})
        with self.assertRaises(ValueError):
            FlagConstraints.from_dict({"exclusive": ["-O2", "-O3"]})

    def test_validate_against_reports_unknown_flags(self):
        c = _sample_constraints()
        warnings = c.validate_against({"-O2", "-O3", "-funroll-loops"})
        self.assertTrue(any("-fno-unroll-loops" in w for w in warnings))
        self.assertEqual(c.validate_against({
            "-O2", "-O3", "-funroll-loops", "-fno-unroll-loops",
            "-fstrict-aliasing", "-fno-strict-aliasing", "-ffast-math",
            "-fno-math-errno", "-ffp-contract=fast",
        }), [])

    def test_expand_pool_adds_exclusive_members(self):
        c = _sample_constraints()
        expanded = c.expand_pool(["-ffast-math"])
        self.assertEqual(expanded[0], "-ffast-math")
        for flag in ("-funroll-loops", "-fno-unroll-loops",
                     "-fstrict-aliasing", "-fno-strict-aliasing"):
            self.assertIn(flag, expanded)
        self.assertEqual(len(expanded), len(set(expanded)))
        # implies values are NOT added implicitly
        self.assertNotIn("-fno-math-errno", expanded)
        self.assertNotIn("-ffp-contract=fast", expanded)

    def test_can_add_exclusive_ignores_implies(self):
        c = _sample_constraints()
        # Implied: blocked as a hard constraint, allowed as a (redundant) node.
        self.assertFalse(c.can_add("-funroll-loops", ["-O3"]))
        self.assertTrue(c.can_add_exclusive("-funroll-loops", ["-O3"]))
        # Exclusive conflict is blocked either way.
        self.assertFalse(c.can_add_exclusive("-funroll-loops", ["-fno-unroll-loops"]))

    def test_sanitize_exclusive_keeps_implied_flags(self):
        c = _sample_constraints()
        self.assertEqual(
            c.sanitize_exclusive(["-O3", "-funroll-loops", "-fno-unroll-loops"]),
            ["-O3", "-funroll-loops"],
        )

    def test_effective_signature_removes_implied_only(self):
        c = _sample_constraints()
        # Implied flags collapse into the implying flag ...
        self.assertEqual(c.effective_signature(["-O3", "-funroll-loops"]), ("-O3",))
        # ... but an exclusive conflict is never silently collapsed, otherwise two
        # genuinely different configurations could look equivalent and one would
        # be skipped.
        self.assertEqual(
            c.effective_signature(["-funroll-loops", "-fno-unroll-loops"]),
            ("-funroll-loops", "-fno-unroll-loops"),
        )


class FlagConstraintsOptunaTests(unittest.TestCase):
    def test_implied_pool_flag_is_discarded(self):
        trial = StubTrial({"flag_variant": "-O3", "pool_k": 3})
        key, _flags = suggest_compiler_flags(
            trial,
            compiler_flags_base="",
            variants=["-O2", "-O3"],
            params_schema={},
            flag_pool=["-funroll-loops", "-fno-unroll-loops", "-ffast-math", "-ffp-contract=fast"],
            compiler_params_select={"min": 0, "max": 0},
            n_startup_trials=10,
            constraints=_sample_constraints(),
        )
        tokens = set(key.split("|"))
        self.assertNotIn("-funroll-loops", tokens)      # implied by -O3
        self.assertNotIn("-ffp-contract=fast", tokens)  # implied by -ffast-math
        self.assertIn("-ffast-math", tokens)
        self.assertIn("-fno-unroll-loops", tokens)      # override still testable

    def test_exclusive_pool_pair_is_never_co_selected(self):
        trial = StubTrial({"flag_variant": "-O2", "pool_k": 4})
        key, _flags = suggest_compiler_flags(
            trial,
            compiler_flags_base="",
            variants=["-O2"],
            params_schema={},
            flag_pool=[
                "-funroll-loops", "-fno-unroll-loops",
                "-fstrict-aliasing", "-fno-strict-aliasing",
            ],
            compiler_params_select={"min": 0, "max": 0},
            n_startup_trials=10,
            constraints=_sample_constraints(),
        )
        tokens = set(key.split("|"))
        self.assertFalse({"-funroll-loops", "-fno-unroll-loops"} <= tokens)
        self.assertFalse({"-fstrict-aliasing", "-fno-strict-aliasing"} <= tokens)


class FlagConstraintsConfigTests(unittest.TestCase):
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
            "compiler_flags": ["-O2", "-O3"],
            "compiler_flag_pool": [
                "-funroll-loops", "-fno-unroll-loops",
                "-ffast-math", "-ffp-contract=fast",
            ],
            "env": {"MODE": ["a"]},
            "objectives": [{"metric": "CPI", "goal": "min"}],
            "perf": {"events": ["cycles"]},
            "search": {"study": "optuna", "sampler": "rs"},
        }
        raw.update(overrides)
        path = self.root / "config.json"
        path.write_text(json.dumps(raw))
        return path

    def test_flag_constraints_are_parsed(self):
        cfg = Config.load(self.write_config(flag_constraints={
            "exclusive": [["-funroll-loops", "-fno-unroll-loops"]],
            "implies": {"-O3": ["-funroll-loops"]},
        }))
        self.assertFalse(cfg.flag_constraints.is_empty)
        self.assertTrue(cfg.flag_constraints.conflicts("-funroll-loops", ["-fno-unroll-loops"]))
        self.assertEqual(cfg.flag_constraints.implied_flags(["-O3"]), frozenset({"-funroll-loops"}))

    def test_absent_flag_constraints_default_to_empty(self):
        cfg = Config.load(self.write_config())
        self.assertTrue(cfg.flag_constraints.is_empty)

    def test_exclusive_flags_become_implicit_pool_members(self):
        cfg = Config.load(self.write_config(
            compiler_flag_pool=["-ffast-math"],
            flag_constraints={"exclusive": [["-funroll-loops", "-fno-unroll-loops"]]},
        ))
        self.assertIn("-ffast-math", cfg.compiler_flag_pool)
        self.assertIn("-funroll-loops", cfg.compiler_flag_pool)
        self.assertIn("-fno-unroll-loops", cfg.compiler_flag_pool)

    def test_unknown_reference_loads_but_is_not_fatal(self):
        cfg = Config.load(self.write_config(flag_constraints={
            "implies": {"-O3": ["-not-in-the-pool"]},
        }))
        self.assertIn("-O3", cfg.flag_constraints.implies)

    def test_malformed_flag_constraints_are_rejected(self):
        with self.assertRaises(ValueError):
            Config.load(self.write_config(flag_constraints={"exclusive": [["-O2"]]}))
        with self.assertRaises(ValueError):
            Config.load(self.write_config(flag_constraints={"unknown_key": []}))


if __name__ == "__main__":
    unittest.main()
