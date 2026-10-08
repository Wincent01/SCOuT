# src/studies/beam_tabu.py
from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import csv, json, math, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable, Union
import tempfile
import os

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import Config
from src.evaluator import Evaluator
from src.misc import unique_csv_path, is_significant_improvement, rel_gain
from src.logger import LOG

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
MetricDict = Dict[str, Number]

# ---------- ENV ENUMERATION (same semantics as in Wavefront) ----------

def _enumerate_env_schema(schema: Dict[str, Union[List[str], Dict[str, Any]]]) -> List[Dict[str, str]]:
    keys = list(schema.keys())
    results: List[Dict[str, str]] = []

    def rec(i: int, partial: Dict[str, str]):
        if i == len(keys):
            results.append(dict(partial)); return
        var = keys[i]; spec = schema[var]
        if isinstance(spec, list):
            for v in spec:
                partial[var] = str(v); rec(i+1, partial)
            partial.pop(var, None); return
        if isinstance(spec, dict) and "values" in spec:
            pred = spec.get("when", {})
            if all(partial.get(k) == str(v) for k,v in pred.items()):
                for v in spec["values"]:
                    partial[var] = str(v); rec(i+1, partial)
                partial.pop(var, None)
            else:
                rec(i+1, partial)
            return
        rec(i+1, partial)

    rec(0, {})
    uniq = {tuple(sorted(d.items())): d for d in results}
    return list(uniq.values())

def _env_combos(cfg: Config, mode: str, cap: Optional[int], rng: random.Random, fixed_env: Dict[str,str]) -> List[Dict[str,str]]:
    mode = (mode or "product").lower()
    if mode == "fixed":
        return [dict(fixed_env or {})]
    schema = getattr(cfg, "env", {}) or {}
    combos = _enumerate_env_schema(schema)
    if not combos: return [{}]
    if not cap or cap <= 0 or len(combos) <= cap: return combos
    rng.shuffle(combos)
    return combos[:cap]  # deterministic slice after shuffle

# ---------- PARAMS ----------

@dataclass
class _BTParams:
    base_flags: List[str] = field(default_factory=list)
    flag_atoms: Optional[List[str]] = None
    max_k: int = 3
    beam_width: int = 16
    per_iter_cap: Optional[int] = None
    env_mode: str = "product"
    env_cap: Optional[int] = None
    env: Dict[str, str] = field(default_factory=dict)
    stop_if_no_improve: bool = True
    improvement_eps: float = 0.0

@dataclass
class _TabuParams:
    max_iters: int = 50
    max_no_improve: int = 5
    tabu_tenure: int = 25
    neighborhood: int = 24
    allow_add_moves: bool = True
    allow_del_moves: bool = True
    results_csv: Optional[str] = None

def _bt_params_from_cfg(cfg: Config) -> _BTParams:
    raw = getattr(cfg, "beam_tabu", {}) or {}
    p = _BTParams()
    for k in p.__dataclass_fields__.keys():
        if k in raw and raw[k] is not None:
            setattr(p, k, raw[k])
    return p

def _tabu_params_from_cfg(cfg: Config) -> _TabuParams:
    raw = getattr(cfg, "tabu", {}) or {}
    p = _TabuParams()
    for k in p.__dataclass_fields__.keys():
        if k in raw and raw[k] is not None:
            setattr(p, k, raw[k])
    return p

# ---------- OBJECTIVE & ATOMS ----------

def _choose_objective(cfg: Config) -> Tuple[str, str]:
    obj = cfg.objectives[0]
    return obj.metric, ("min" if obj.goal == "min" else "max")

def _collect_atoms(cfg: Config, p: _BTParams) -> List[str]:
    if p.flag_atoms:
        atoms = list(dict.fromkeys(p.flag_atoms))
    else:
        atoms: List[str] = []
        pool = getattr(cfg, "compiler_flag_pool", None)
        if pool: atoms.extend(list(pool))
        cf = getattr(cfg, "compiler_flags", None)
        if isinstance(cf, list): atoms.extend(cf)
        elif isinstance(cf, dict): atoms.extend(cf.values())
        atoms = list(dict.fromkeys(atoms))
    if not atoms:
        raise ValueError("beam_tabu: no flag atoms found.")
    return atoms

# ---------- BUILD & MEASURE ----------

def _compile_and_measure(evaluator: Evaluator, cfg: Config, flags: Sequence[str], env: Dict[str, str], work: Path
) -> Tuple[float, MetricDict, str]:
    evaluation = evaluator.evaluate(flags, env, work)
    metrics = evaluation.metrics
    binary = evaluation.binary

    metric_name, goal = _choose_objective(cfg)
    if metric_name not in metrics:
        raise RuntimeError(f"objective metric '{metric_name}' missing; got: {list(metrics.keys())}")
    value = float(metrics[metric_name])
    return value, metrics, str(binary)

# ---------- NEIGHBORHOOD & TABU ----------

def _canonical(flags: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(flags)))

def _neighbors(parent: Tuple[str,...], atoms: List[str], max_k: int, allow_add: bool, allow_del: bool, limit: int,
               constraints=None, base_flags: Sequence[str] = ()) -> List[Tuple[str,...]]:
    present = set(parent)
    additions: List[Tuple[str,...]] = []
    deletions: List[Tuple[str,...]] = []

    # add moves
    if allow_add and len(parent) < max_k:
        active = list(base_flags) + list(parent)
        for a in atoms:
            if a in present:
                continue
            if constraints is not None and not constraints.is_empty:
                if not constraints.can_add(a, active):
                    continue
            additions.append(_canonical(parent + (a,)))
    # delete moves
    if allow_del and len(parent) > 0:
        for a in parent:
            c = list(parent); c.remove(a)
            deletions.append(_canonical(c))

    cand: List[Tuple[str, ...]] = []
    for i in range(max(len(additions), len(deletions))):
        if i < len(additions):
            cand.append(additions[i])
        if i < len(deletions):
            cand.append(deletions[i])

    # dedup and return
    uniq = []
    seen = set()
    for t in cand:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq[:limit] if limit and limit > 0 else uniq

def _move_from_to(src: Tuple[str,...], dst: Tuple[str,...]) -> Tuple[str,str]:
    s, d = set(src), set(dst)
    add = list(d - s)
    rem = list(s - d)
    if add: return ("add", add[0])
    if rem: return ("del", rem[0])
    return ("noop","")

# ---------- MAIN STUDY ----------

def run_beam_tabu_study(cfg: Config) -> None:
    bt = _bt_params_from_cfg(cfg)
    tb = _tabu_params_from_cfg(cfg)

    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))
    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_beamtabu_"))

    # If we are in a cluster environment with a TMPDIR, use that as the workroot instead of /tmp
    if (os.getenv("TMPDIR")):
        workroot = Path(os.getenv("TMPDIR")) / workroot.name
    
    evaluator = Evaluator(cfg, workroot)
    atoms = _collect_atoms(cfg, bt)
    constraints = getattr(cfg, "flag_constraints", None)
    metric_name, goal = _choose_objective(cfg)
    sig = cfg.significance
    MIN_REL = sig.min_rel_gain
    MIN_ABS = sig.min_abs_gain

    def score(v: float) -> float: return v if goal == "min" else -v

    # env combos
    env_combos = _env_combos(cfg, bt.env_mode, bt.env_cap, rng, fixed_env=bt.env)
    LOG.log(f"[beam-tabu] workdir={workroot} atoms={len(atoms)} beam={bt.beam_width} iters={tb.max_iters} env_combos={len(env_combos)}")

    # Initial candidate = base_flags (atoms apply *on top* of this)
    base = tuple(dict.fromkeys(
        ([cfg.compiler_flags_base] if cfg.compiler_flags_base else []) + list(bt.base_flags or [])
    ))
    # Evaluate baseline over envs (take best env outcome for ranking)
    best_env_val = 0.0
    best_env_score = math.inf
    best_env_metrics: Dict[str,float] = {}
    best_env_bin = ""
    best_env_env: Dict[str,str] = {}

    # Buffer all rows to write Optuna-like CSV (with leading k)
    rows: List[Tuple[int, List[float], str, Dict[str,str], str, Dict[str,float]]] = []
    extra_metric_keys: set[str] = set()

    for ei, env in enumerate(env_combos, 1):
        try:
            val, mets, binp = _compile_and_measure(evaluator, cfg, list(base), env, workroot / "iter00_baseline" / f"env{ei:03d}")
        except Exception as exc:
            print(f"[beam-tabu] baseline env {ei} failed: {exc}")
            continue
        rows.append((0, [val], "|".join(base) if base else "default", dict(env), str(binp), mets))
        extra_metric_keys.update(mets.keys())
        if score(val) < best_env_score:
            best_env_score = score(val)
            best_env_val, best_env_metrics, best_env_bin, best_env_env = val, mets, binp, env

    if best_env_score == math.inf:
        raise RuntimeError("beam-tabu: all baseline environment evaluations failed")

    best_global_val = best_env_val
    best_global_combo: Tuple[str,...] = tuple()

    # Beam frontier holds tuples of *atom* sets added on top of base (for clarity we treat full string set)
    beam: List[Tuple[str,...]] = [tuple()]  # empty tuple means no atom yet (just base)
    tabu: Dict[Tuple[str,str], int] = {}    # move -> expire_iter
    no_improve = 0

    # Iterations
    for it in range(1, tb.max_iters + 1):
        print(f"[beam-tabu] === iter {it} ===")
        previous_beam = list(beam)
        next_candidates: List[Tuple[float, float, Tuple[str,...], Dict[str,float], str, Dict[str,str]]] = []
        # (score_for_rank, value, combo_atoms, metrics, bin, env_used)

        # Expand each parent in beam
        for parent_atoms in beam:
            # Full current flags = base + parent_atoms
            parent_flags = list(base) + list(parent_atoms)
            neigh = _neighbors(parent_atoms, atoms, bt.max_k, tb.allow_add_moves, tb.allow_del_moves, tb.neighborhood,
                               constraints=constraints, base_flags=list(base))
            # Optional cap per iteration for cost control
            if bt.per_iter_cap and len(neigh) > bt.per_iter_cap:
                rng.shuffle(neigh); neigh = neigh[:bt.per_iter_cap]

            for child_atoms in neigh:
                move = _move_from_to(parent_atoms, child_atoms)
                # Tabu check
                is_tabu = False
                if move[0] != "noop" and move in tabu and tabu[move] > it:
                    is_tabu = True

                flags = list(base) + list(child_atoms)

                # Evaluate across envs, pick best env outcome
                best_c_val = 0.0
                best_c_score = math.inf
                best_c_mets: Dict[str,float] = {}
                best_c_bin = ""
                best_c_env: Dict[str,str] = {}

                for ei, env in enumerate(env_combos, 1):
                    try:
                        val, mets, binp = _compile_and_measure(
                            evaluator, cfg, flags, env,
                            workroot / f"iter{it:02d}" / f"candidate_{len(rows):05d}" / f"env{ei:03d}",
                        )
                    except Exception as exc:
                        print(f"[beam-tabu] candidate failed: {exc}")
                        continue
                    # buffer row for CSV
                    rows.append((it, [val], "|".join(flags) if flags else "default", dict(env), str(binp), mets))
                    extra_metric_keys.update(mets.keys())
                    if score(val) < best_c_score:
                        best_c_score = score(val)
                        best_c_val, best_c_mets, best_c_bin, best_c_env = val, mets, binp, env

                if best_c_score == math.inf:
                    continue

                # Aspiration: allow tabu if it beats best_global
                if is_tabu and score(best_c_val) >= score(best_global_val):
                    continue

                next_candidates.append((score(best_c_val), best_c_val, child_atoms, best_c_mets, best_c_bin, best_c_env))

        if not next_candidates:
            print("[beam-tabu] no candidates; stopping.")
            break

        # Deduplicate children reached from multiple parents, then select the beam.
        by_child = {}
        for candidate in next_candidates:
            child = candidate[2]
            if child not in by_child or candidate[0] < by_child[child][0]:
                by_child[child] = candidate
        next_candidates = list(by_child.values())

        # Sort and select beam
        next_candidates.sort(key=lambda t: t[0])
        selected = next_candidates[: bt.beam_width]

        # Generation best (by objective value)
        gen_best_score, gen_best_val, gen_best_atoms, gen_best_mets, gen_best_bin, gen_best_env = selected[0]

        better = gen_best_score < score(best_global_val)
        generation_gain = rel_gain(best_global_val, gen_best_val, goal)
        meaningful = is_significant_improvement(
            old=best_global_val, new=gen_best_val, goal=goal,
            min_rel_gain=MIN_REL,
            min_abs_gain=max(float(MIN_ABS or 0.0), float(bt.improvement_eps)),
        )
        if better:
            best_global_val = gen_best_val
            best_global_combo = gen_best_atoms
        if better and not meaningful:
            print(f"[beam-tabu] gen best not significant "
                f"(Δrel={generation_gain:.3f})")

        # Build next beam frontier (just the atom-sets)
        beam = [atoms_c for (_sc, _v, atoms_c, _m, _b, _e) in selected]

        # Update tabu memory using parents from the previous generation.
        new_beam: List[Tuple[str,...]] = []
        for sc, val, atoms_c, mets_c, bin_c, env_c in selected:
            new_beam.append(atoms_c)
            # Add move to tabu list (from closest parent — approximated using set diff)
            # Store the move that produced atoms_c from *some* parent in previous beam
            # Here we pick the smallest diff parent to define the move.
            move = None
            best_diff = 1e9
            for parent_atoms in previous_beam:
                diff = len(set(atoms_c) ^ set(parent_atoms))
                if diff < best_diff:
                    best_diff = diff
                    move = _move_from_to(parent_atoms, atoms_c)
            if move and move[0] != "noop":
                reverse = ("del" if move[0] == "add" else "add", move[1])
                tabu[reverse] = it + tb.tabu_tenure

        beam = new_beam

        # Expire tabu entries
        expired = [mv for mv, exp in tabu.items() if exp <= it]
        for mv in expired:
            tabu.pop(mv, None)

        if meaningful:
            no_improve = 0
        else:
            no_improve += 1
            if bt.stop_if_no_improve and no_improve >= tb.max_no_improve:
                print("[beam-tabu] early stop: no improvement streak.")
                break

    # ----- Write CSV like Optuna + Wavefront (with leading 'k') -----
    results_path: Path
    if tb.results_csv:
        results_path = unique_csv_path(tb.results_csv)
    elif getattr(cfg, "csv_log", None):
        results_path = unique_csv_path(cfg.csv_log)
    else:
        results_path = workroot / "beam_tabu_results.csv"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[beam-tabu] writing CSV → {results_path}")

    obj_headers = [o.metric for o in cfg.objectives]
    extra_cols = sorted(extra_metric_keys - set(obj_headers))
    header = ["k"] + obj_headers + ["compiler_flags", "env", "binary"] + extra_cols

    with open(results_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for k_iter, obj_vals, flags_key, env_row, binary_path, metrics in rows:
            if not obj_vals or math.isinf(obj_vals[0]):  # skip failures
                continue
            row = [k_iter] + list(obj_vals) + [flags_key, json.dumps(env_row), binary_path]
            row += [metrics.get(col, "") for col in extra_cols]
            w.writerow(row)

    # Summary
    print("\n[beam-tabu] ===== Summary =====")
    print(f"best {metric_name} = {best_global_val:.6g}")
    print(f"flags = {list(base) + list(best_global_combo)}")
    print(f"[beam-tabu] results → {results_path}")
    print(f"[beam-tabu] unique builds: {evaluator.build_count}")
