from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################
import csv
import json
import math
import random
import tempfile
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Any, Union


###############################################################################
# Type helpers                                                                #
###############################################################################
Number = float
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################
from src.config import Config, ParserConfig, BuildProject
from src.build import compile_project, compile_single_source, _run
from src.metrics import measure_likwid, measure_perf
from src.misc import unique_csv_path, clear_acpp_runtime_cache, is_significant_improvement, rel_gain, save_run_output
from src.evaluator import Evaluator
from src.logger import LOG

@dataclass
class _WFParams:
    # Flags that are always present (baseline)
    base_flags: List[str] = field(default_factory=list)
    # Flag atoms to combine; if None, we will try to infer from config.
    flag_atoms: Optional[List[str]] = None
    # Max combination size (k). Beware C(n,k) growth.
    max_k: int = 3
    # Search flavor:
    #  - "full": evaluate ALL C(n,k) at each level k
    #  - "beam": expand only the top-N combos from previous level
    mode: str = "beam"
    beam_width: int = 16
    # Optional cap on the number of candidates per wave (after generation); None = no cap
    per_wave_cap: Optional[int] = None
    # Stop early if no improvement vs best-so-far by at least eps
    stop_if_no_improve: bool = True
    improvement_eps: float = 0.0
    # Environment for all runs in this study (constant)
    env: Dict[str, str] = field(default_factory=dict)
    env_mode: str = "product"       # "fixed" | "product" | "sample"
    env_cap: Optional[int] = None   # max env combos per candidate (only for product/sample)

    # Output
    results_csv: Optional[str] = None


def _enumerate_env_schema(schema: Dict[str, Union[List[str], Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Enumerate all consistent env assignments honoring 'when' predicates.
    Dict insertion order is preserved (Python 3.7+), so place dependents after the vars they depend on.
    Example entry formats:
      VAR: ["a","b"]                                  -> unconditional choices
      VAR: {"when": {"OTHER":"x"}, "values":[...]}    -> only assign VAR if predicate matches current partial env
    """
    keys = list(schema.keys())

    def rec(i: int, partial: Dict[str, str], out: List[Dict[str, str]]):
        if i == len(keys):
            out.append(dict(partial))
            return
        var = keys[i]
        spec = schema[var]

        # Unconditional list
        if isinstance(spec, list):
            for v in spec:
                partial[var] = str(v)
                rec(i + 1, partial, out)
            partial.pop(var, None)
            return

        # Conditional object
        if isinstance(spec, dict) and "values" in spec:
            pred = spec.get("when", {})
            # only if all predicates match the *already-assigned* vars
            if all(partial.get(k) == str(v) for k, v in pred.items()):
                for v in spec["values"]:
                    partial[var] = str(v)
                    rec(i + 1, partial, out)
                partial.pop(var, None)
            else:
                # Predicate not satisfied now → skip assigning this var
                rec(i + 1, partial, out)
            return

        # Unknown format → ignore this key (robust)
        rec(i + 1, partial, out)

    results: List[Dict[str, str]] = []
    rec(0, {}, results)
    # Deduplicate just in case
    uniq = {tuple(sorted(d.items())): d for d in results}
    return list(uniq.values())


def _env_combos_for_wavefront(
    cfg: "Config",
    params: "_WFParams",
    rng: random.Random,
) -> List[Dict[str, str]]:
    """
    Decide which envs to evaluate:
      - "fixed": use params.env (single combo)
      - "product": enumerate all consistent combos from cfg.env (honor 'when'); cap via env_cap
      - "sample": like product but randomly downsample to env_cap
    """
    mode = (getattr(params, "env_mode", "product") or "product").lower()
    if mode == "fixed":
        return [dict(params.env or {})]

    # Build from the root config's env schema
    schema = getattr(cfg, "env", {}) or {}
    combos = _enumerate_env_schema(schema)
    if not combos:
        return [{}]

    cap = getattr(params, "env_cap", None)
    if cap is None or cap <= 0 or len(combos) <= cap:
        return combos

    # Need to reduce
    if mode == "product":
        # deterministic prefix slice after shuffle for diversity across runs
        rng.shuffle(combos)
        return combos[:cap]
    if mode == "sample":
        return rng.sample(combos, cap)
    return combos  # fallback


def _params_from_cfg(cfg: Config) -> _WFParams:
    wf = getattr(cfg, "wavefront", None)
    p = _WFParams()
    if not wf:
        return p

    # If user hasn’t rebuilt Config yet and wf is still a dict, handle it.
    if isinstance(wf, dict):
        for k in p.__dataclass_fields__.keys():
            if k in wf and wf[k] is not None:
                setattr(p, k, wf[k])
        return p

    # Normal path: WavefrontSpec dataclass
    for k in p.__dataclass_fields__.keys():
        if hasattr(wf, k):
            setattr(p, k, getattr(wf, k))
    return p


def _choose_objective(cfg: Config) -> Tuple[str, str]:
    """Return (metric_name, goal) with goal in {'min','max'}; use first objective."""
    if not cfg.objectives:
        raise ValueError("No objectives in config.")
    obj = cfg.objectives[0]
    return obj.metric, ("min" if obj.goal == "min" else "max")


def _collect_atoms(cfg: Config, p: _WFParams) -> List[str]:
    if p.flag_atoms:
        atoms = list(dict.fromkeys(p.flag_atoms))
    else:
        atoms: List[str] = []
        pool = getattr(cfg, "compiler_flag_pool", None)
        if pool: atoms.extend(list(pool))
        cf = getattr(cfg, "compiler_flags", None)
        if isinstance(cf, list):
            atoms.extend(cf)
        elif isinstance(cf, dict):
            atoms.extend(cf.values())
        atoms = list(dict.fromkeys(atoms))
    if not atoms:
        raise ValueError("wavefront: no flag atoms. Provide wavefront.flag_atoms or populate compiler_flag_pool/compiler_flags.")
    return atoms


def _compile_and_measure(evaluator: Evaluator, cfg: Config, flags: Sequence[str], env: Dict[str, str], work: Path
) -> Tuple[float, MetricDict, str]:
    evaluation = evaluator.evaluate(flags, env, work)
    metrics = evaluation.metrics
    binary = evaluation.binary

    metric_name, goal = _choose_objective(cfg)
    if metric_name not in metrics:
        raise RuntimeError(f"objective metric '{metric_name}' missing; got: {list(metrics.keys())}")
    return float(metrics[metric_name]), metrics, str(binary)


def _score(value: float, goal: str) -> float:
    """Lower is always better for internal comparisons."""
    return value if goal == "min" else -value


def _canonical(flags: Iterable[str]) -> Tuple[str, ...]:
    """Order-insensitive canonical tuple (sorted unique) to dedup combos."""
    # Keep deterministic ordering by string sort; adapt if flag order matters in your toolchain.
    return tuple(sorted(dict.fromkeys(flags)))


def _generate_full(atoms: List[str], k: int) -> Iterable[Tuple[str, ...]]:
    for combo in combinations(atoms, k):
        yield _canonical(combo)


def _generate_beam(expand_from: List[Tuple[str, ...]], atoms: List[str]) -> Iterable[Tuple[str, ...]]:
    seen: set[Tuple[str, ...]] = set()
    for combo in expand_from:
        for a in atoms:
            if a in combo:
                continue
            newc = _canonical(combo + (a,))
            if newc not in seen:
                seen.add(newc)
                yield newc

def _generate_beam_boost(
    expand_from: List[Tuple[str, ...]],
    atoms: List[str],
    rng: random.Random,
    focus_count: int = 1
) -> Iterable[Tuple[str, ...]]:

    seen: set[Tuple[str, ...]] = set()

    # Make local copies so we don't mutate originals
    atoms_local = list(atoms)
    rng.shuffle(atoms_local)

    # 1) Focus parents first (deterministic, already ordered by score)
    for combo in expand_from[:max(0, focus_count)]:
        for a in atoms_local:
            if a in combo:
                continue
            newc = _canonical(combo + (a,))
            if newc not in seen:
                seen.add(newc)
                yield newc

    # 2) Then the rest, in random parent order
    tail = list(expand_from[max(0, focus_count):])
    rng.shuffle(tail)
    for combo in tail:
        for a in atoms_local:
            if a in combo:
                continue
            newc = _canonical(combo + (a,))
            if newc not in seen:
                seen.add(newc)
                yield newc



_SYCL_RE = re.compile(
    r'^\[SYCL\]\[(?P<label>avg|sum)\]\s*kernel\s*(?P<kid>\d+)\s*:\s*'
    r'(?P<val>[0-9]*\.?[0-9]+)\s*s\s*over\s*(?P<iters>\d+)\s*iters\s*$',
    re.IGNORECASE | re.MULTILINE
)

_GROMACS_PERF_RE = re.compile(
    r"Performance:\s+"
    r"(?P<ns_per_day>[0-9]*\.?[0-9]+)\s+"
    r"(?P<hour_per_ns>[0-9]*\.?[0-9]+)\s+"
    r"(?P<ms_per_step>[0-9]*\.?[0-9]+)\s+"
    r"(?P<matom_steps_per_s>[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

def _wf_resolve_cwd(run_cwd: str, bin_path: Path, workdir: Optional[Path], project: Optional[BuildProject]) -> Path:
    if run_cwd == "workdir" and workdir:
        return workdir
    if run_cwd == "project_dir" and project:
        return project.dir
    return bin_path.parent  # "binary_dir"

def _wf_aggregate(vals: List[float], how: str) -> float:
    how = (how or "sum").lower()
    if how == "mean": return float(mean(vals))
    if how == "max":  return float(max(vals))
    if how == "min":  return float(min(vals))
    return float(sum(vals))  # default sum

def _tail_excerpt(stdout: str, stderr: str, n: int = 20) -> str:
    """Return the last n lines of a run's merged output (for failure messages)."""
    text = (stdout or "") + (("\n" + stderr) if stderr else "")
    tail = text.strip().splitlines()[-n:]
    return "\n".join(tail) if tail else "(no output)"

def measure_parser_sycl_wavefront(
    pcfg: ParserConfig,
    bin_path: Path,
    prog_args: List[str],
    env: Dict[str, str],
    runs: int,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
) -> Dict[str, float]:
    """
    Launch like perf/likwid (prefix/taskset/cwd), parse standardized SYCL lines,
    discard warm-up iterations, and return metrics dict.

    Emits:
      - per kernel:   sycl_kernel_<id>_<label>_s
      - aggregate:    sycl_<label>_<aggregate>_s
      - (optional)    sycl_iters  (if consistent and present)
    """
    merged_env = {**os.environ, **env}

    cmd: List[str] = []
    if pcfg.prefix:
        cmd.extend(pcfg.prefix)
    if pcfg.core_list:
        cmd.extend(["taskset", "-c", pcfg.core_list])
    cmd.append(str(bin_path))
    cmd.extend(prog_args)

    cwd = _wf_resolve_cwd(pcfg.run_cwd, Path(bin_path), workdir, project)

    want_label = (pcfg.label or "avg").lower()
    parse_format = (getattr(pcfg, "format", "sycl") or "sycl").lower()
    meas_runs = max(1, runs)
    total_runs = int(getattr(pcfg, "warmup_runs", 0)) + meas_runs
    warmup_cut = int(getattr(pcfg, "warmup_runs", 0))

    runs_kernel_vals: List[Dict[int, float]] = []
    iterations_seen: List[int] = []

    for i in range(total_runs):
        proc = _run(cmd, cwd=cwd, env=merged_env)
        tmp = save_run_output(proc.stdout or "", proc.stderr or "", prefix=f"scout_run_{i:02d}_")
        print(f"[run] {i:02d} output → {tmp}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"program exited with rc={proc.returncode}.\n"
                f"  output → {tmp}\n"
                f"  --- tail ---\n{_tail_excerpt(proc.stdout or '', proc.stderr or '')}"
            )

        # ignore parse errors during warmup; still execute the binary to JIT
        if i < warmup_cut:
            continue

        text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        if parse_format == "gromacs":
            perf_matches = list(_GROMACS_PERF_RE.finditer(text))
            if not perf_matches:
                logs = (workdir or cwd) / "parser_logs"
                try:
                    logs.mkdir(parents=True, exist_ok=True)
                    (logs / f"no_match_{i:02d}.out").write_text(proc.stdout or "")
                    (logs / f"no_match_{i:02d}.err").write_text(proc.stderr or "")
                except Exception:
                    pass
                raise RuntimeError(
                    "Parser backend (GROMACS): no matching Performance line found.\n"
                    f"  output → {tmp}\n"
                    f"  --- tail ---\n{_tail_excerpt(proc.stdout or '', proc.stderr or '')}"
                )
            m = perf_matches[-1]
            runs_kernel_vals.append({
                0: float(m.group("ns_per_day")),
                1: float(m.group("hour_per_ns")),
                2: float(m.group("ms_per_step")),
                3: float(m.group("matom_steps_per_s")),
            })
            iterations_seen.append(-1)
            continue

        per_kernel: Dict[int, float] = {}
        iters_val: Optional[int] = None

        for m in _SYCL_RE.finditer(text):
            label = m.group("label").lower()
            if label != want_label:
                continue
            kid   = int(m.group("kid"))
            val   = float(m.group("val"))    # seconds
            iters = int(m.group("iters"))
            iters_val = iters
            per_kernel[kid] = val

        if not per_kernel:
            # Helpful dump for debugging
            logs = (workdir or cwd) / "parser_logs"
            try:
                logs.mkdir(parents=True, exist_ok=True)
                (logs / f"no_match_{i:02d}.out").write_text(proc.stdout or "")
                (logs / f"no_match_{i:02d}.err").write_text(proc.stderr or "")
            except Exception:
                pass
            raise RuntimeError(
                "Parser backend (SYCL): no matching [SYCL] lines found.\n"
                f"  output → {tmp}\n"
                f"  --- tail ---\n{_tail_excerpt(proc.stdout or '', proc.stderr or '')}"
            )

        runs_kernel_vals.append(per_kernel)
        iterations_seen.append(iters_val if iters_val is not None else -1)

    if parse_format == "gromacs":
        ns_per_day_vals = [d[0] for d in runs_kernel_vals if 0 in d]
        hour_per_ns_vals = [d[1] for d in runs_kernel_vals if 1 in d]
        ms_per_step_vals = [d[2] for d in runs_kernel_vals if 2 in d]
        matom_steps_vals = [d[3] for d in runs_kernel_vals if 3 in d]
        if not ns_per_day_vals:
            raise RuntimeError("Parser backend (GROMACS): no throughput values collected.")

        LOG.write(f"[parser] gmx_ns_per_day = {mean(ns_per_day_vals):.6g}")
        LOG.write(f"[parser] gmx_hour_per_ns = {mean(hour_per_ns_vals):.6g}")
        LOG.write(f"[parser] gmx_ms_per_step = {mean(ms_per_step_vals):.6g}")
        LOG.write(f"[parser] gmx_matom_steps_per_s = {mean(matom_steps_vals):.6g}")

        clear_acpp_runtime_cache()
        return {
            "gmx_ns_per_day": float(mean(ns_per_day_vals)),
            "gmx_hour_per_ns": float(mean(hour_per_ns_vals)) if hour_per_ns_vals else 0.0,
            "gmx_ms_per_step": float(mean(ms_per_step_vals)) if ms_per_step_vals else 0.0,
            "gmx_matom_steps_per_s": float(mean(matom_steps_vals)) if matom_steps_vals else 0.0,
        }

    # average across measured runs per kernel
    all_kids = sorted({k for d in runs_kernel_vals for k in d.keys()})
    # filter if user specified explicit kernel IDs
    selected = all_kids
    if getattr(pcfg, "kernels", None):
        ks = set(int(x) for x in pcfg.kernels)
        selected = [k for k in all_kids if k in ks]

    per_kernel_avg: Dict[int, float] = {}
    for k in selected:
        vals = [d[k] for d in runs_kernel_vals if k in d]
        if vals:
            per_kernel_avg[k] = float(mean(vals))

    if not per_kernel_avg:
        raise RuntimeError("Parser backend (SYCL): selected kernels missing in output.")

    # aggregate across selected kernels
    aggregate_val = _wf_aggregate(list(per_kernel_avg.values()), pcfg.aggregate)

    # build metrics dict
    mets: Dict[str, float] = {}
    for k, v in per_kernel_avg.items():
        mets[f"sycl_kernel_{k}_{want_label}_s"] = v
    mets[f"sycl_{want_label}_{pcfg.aggregate}_s"] = aggregate_val

    if iterations_seen and all(i == iterations_seen[0] and i >= 0 for i in iterations_seen):
        mets["sycl_iters"] = float(iterations_seen[0])

    clear_acpp_runtime_cache()
    return mets

def run_wavefront_study(cfg: Config) -> None:
    """
    Wave-front search over flag atoms:
      k=0: baseline (base_flags)
      k=1: all singles
      k=2: all pairs (or beam expansions)
      ...
    Logs CSV and prints best per wave.
    """
    params = _params_from_cfg(cfg)
    atoms = _collect_atoms(cfg, params)
    metric_name, goal = _choose_objective(cfg)

    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))
    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_wave_"))
    evaluator = Evaluator(cfg, workroot)
    print(f"[wavefront] workdir: {workroot}")
    print(f"[wavefront] atoms={len(atoms)} max_k={params.max_k} mode={params.mode} beam_width={params.beam_width}")
    if params.per_wave_cap:
        print(f"[wavefront] per_wave_cap={params.per_wave_cap}")

    base_env: Dict[str, str] = dict(getattr(getattr(cfg, "wavefront", object()), "env", {}) or {})
    env_combos = _env_combos_for_wavefront(cfg, params, rng)
    print(f"[wavefront] env_mode={params.env_mode} env_combos={len(env_combos)}")

    base_flags = list(dict.fromkeys(
        ([cfg.compiler_flags_base] if cfg.compiler_flags_base else []) + list(params.base_flags or [])
    ))

    constraints = getattr(cfg, "flag_constraints", None)
    if constraints is not None and not constraints.is_empty:
        # Atoms already implied by the baseline are redundant to test on their own.
        redundant = constraints.implied_flags(base_flags)
        atoms = [a for a in atoms if a not in redundant]

    # Baseline
    print("[wavefront] evaluating baseline …")
    base_dir = workroot / "k00_baseline"
    rows: List[Tuple[int, List[float], str, Dict[str, str], str, Dict[str, float]]] = []
    extra_metric_keys: set[str] = set()

    metric_name, goal = _choose_objective(cfg)
    sig = cfg.significance
    MIN_REL = sig.min_rel_gain
    MIN_ABS = sig.min_abs_gain
    def _score(v: float) -> float: return v if goal == "min" else -v

    best_base_sc = math.inf
    best_base = None

    for ei, env in enumerate(env_combos, 1):
        run_dir = base_dir / f"env{ei:03d}"
        try:
            val, mets, binpath = _compile_and_measure(evaluator, cfg, base_flags, env, run_dir)
        except Exception as exc:
            print(f"[wavefront] baseline env {ei} failed: {exc}")
            continue
        rows.append((0, [val], "|".join(base_flags) or "default", dict(env), str(binpath), mets))
        extra_metric_keys.update(mets.keys())
        sc = _score(val)
        if sc < best_base_sc:
            best_base_sc = sc
            best_base = (val, mets, binpath, env)

    if best_base is None:
        raise RuntimeError("wavefront: all baseline environment evaluations failed.")
    base_val, base_metrics, base_bin, base_env = best_base
    best_global_score = best_base_sc
    best_val_global = base_val
    best_global_combo: Tuple[str, ...] = tuple()
    print(f"[wavefront] baseline {metric_name} = {base_val:.6g} env={json.dumps(base_env)}")


    def _flags_key(flags_seq: Sequence[str]) -> str:
        # match Optuna’s "pretty id" (pipe-joined)
        return "|".join(flags_seq) if flags_seq else "default"

    # Wave k >= 1
    prev_top: List[Tuple[str, ...]] = []
    for k in range(1, params.max_k + 1):
        print(f"[wavefront] ===== Wave k={k} =====")

        # Generate candidates
        if params.mode == "full" or (k == 1 and not prev_top):
            gen_iter = _generate_full(atoms, k)
        else:
            #gen_iter = _generate_beam(prev_top, atoms)
            gen_iter = _generate_beam_boost(prev_top, atoms, rng, focus_count=1)
        candidates = list(gen_iter)
        if constraints is not None and not constraints.is_empty:
            candidates = [
                c for c in candidates
                if constraints.is_allowed(list(base_flags) + list(c))
            ]
        if params.per_wave_cap and len(candidates) > params.per_wave_cap:
            rng.shuffle(candidates)
            candidates = candidates[: params.per_wave_cap]
            candidates.sort()

        print(f"[wavefront] candidates={len(candidates)}")
        if not candidates:
            print("[wavefront] no candidates; stop.")
            break

        # Evaluate all candidates for this wave
        scored: List[Tuple[float, float, Tuple[str, ...], MetricDict, str]] = []
        # tuple: (score_for_beam, best_value_among_envs, combo_flags, best_metrics, best_binary)

        for idx, combo in enumerate(candidates, 1):
            flags = base_flags + list(combo)
            # Evaluate ALL env combos (or reduced set) for this flag combo
            best_sc_c = math.inf
            best_val_c = math.inf
            best_mets_c: Dict[str, float] = {}
            best_bin_c = ""

            for ei, env in enumerate(env_combos, 1):
                run_dir = workroot / f"k{k:02d}" / f"c{idx:05d}" / f"env{ei:03d}"
                try:
                    value, metrics, binary = _compile_and_measure(evaluator, cfg, flags, env, run_dir)
                    sc = _score(value)
                except Exception as exc:
                    value, sc, metrics, binary = (math.inf, math.inf, {"error": str(exc)}, "")

                # buffer a row for Optuna-like CSV (each env eval is a row)
                rows.append((k, [float(value)], "|".join(flags) if flags else "default", dict(env), str(binary), metrics))
                extra_metric_keys.update(metrics.keys())

                # pick best env outcome for ranking this flag combo
                if sc < best_sc_c:
                    best_sc_c, best_val_c, best_mets_c, best_bin_c = sc, value, metrics, binary

            scored.append((best_sc_c, best_val_c, tuple(combo), best_mets_c, best_bin_c))

        # Sort by score (lower better)
        scored.sort(key=lambda t: t[0])
        best_sc, best_val, best_combo, best_metrics, best_bin = scored[0]

        print(f"[wavefront] best@k={k}: {metric_name}={best_val:.6g}  flags={list(best_combo)}")

        # Seeds for next wave (beam)
        if params.mode == "beam":
            prev_top = [c for _sc, _val, c, _m, _b in scored[: params.beam_width]]
        else:
            prev_top = []

        # Early stop if not improving globally
        better = _score(best_val) < _score(best_val_global)
        meaningful = is_significant_improvement(old=best_val_global,
                              new=best_val,
                              goal=goal,
                              min_rel_gain=MIN_REL,
                              min_abs_gain=max(float(MIN_ABS or 0.0), float(params.improvement_eps)))
        if better:
            best_val_global = best_val
            best_global_combo = best_combo
        if not meaningful:
            if params.stop_if_no_improve:
                print("[wavefront] no *significant* improvement; stopping early.")
                break

    # --- Write CSV IDENTICAL to explore_optuna() ---
    if params.results_csv:
        results_path = unique_csv_path(params.results_csv)
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    elif getattr(cfg, "csv_log", None):
        results_path = unique_csv_path(cfg.csv_log)
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        results_path = workroot / "wavefront_results.csv"

    print(f"[wavefront] writing CSV → {results_path}")

    obj_headers = [o.metric for o in cfg.objectives]  # usually 1 metric for wavefront
    extra_cols = sorted(extra_metric_keys - set(obj_headers))
    header = ["k"] + obj_headers + ["compiler_flags", "env", "binary"] + extra_cols

    with open(results_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for k_idx, obj_vals, flags_key, env_row, binary_path, metrics in rows:
            # mimic Optuna: skip failed (value==inf)
            if not obj_vals or math.isinf(obj_vals[0]):
                continue
            row = [k_idx] + list(obj_vals) + [flags_key, json.dumps(env_row), binary_path]
            row += [metrics.get(k, "") for k in extra_cols]
            w.writerow(row)

    print(f"[wavefront] results → {results_path}")
    print(f"[wavefront] best {metric_name}={best_val_global:.6g} flags={base_flags + list(best_global_combo)}")
    print(f"[wavefront] unique builds: {evaluator.build_count}")
