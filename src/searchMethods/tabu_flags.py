from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################
import json
import math
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

###############################################################################
# Type helpers                                                                #
###############################################################################
Number = float
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################
from src.checkpoint import (
    EvalRecord,
    RunJournal,
    StopRequested,
    config_fingerprint,
    eval_id,
    install_sigterm_handler,
    restore_sigterm_handler,
    rng_from_json,
    rng_to_json,
)
from src.config import Config
from src.evaluator import Evaluator
from src.misc import clean_run_outputs, is_significant_improvement



# -----------------------------
# Config block for Tabu Search
# -----------------------------
@dataclass
class TabuSpec:
    # Search limits
    max_iters: int = 200
    max_no_improve: int = 50

    # Tabu list length (tenure)
    tabu_tenure: int = 20

    # Neighborhood sampling per iteration
    neighborhood: int = 24

    # Controls which components are allowed to change in neighbors
    allow_variant_moves: bool = True
    allow_param_moves: bool = True
    allow_pool_moves: bool = True
    allow_env_moves: bool = True

    # Environment exploration (matches wavefront style)
    env_mode: str = "product"         # "fixed" | "product" | "sample"
    env_cap: Optional[int] = None     # cap/compress env combos if large
    env: Dict[str, str] = field(default_factory=dict)

    # CSV file; default will fall back to cfg.csv_log
    results_csv: Optional[str] = None

    # Optional JSON snapshot of the search state (default: next to the CSV)
    state_file: Optional[str] = None
    # Optional failure CSV (default: cfg.failed_builds, else next to the CSV)
    failure_csv: Optional[str] = None
    # Where builds and measurements live (default: $SCOUT_TABU_WORKROOT,
    # then $TMPDIR/SCOuT_tabu). Builds are reused when they are still there.
    run_dir: Optional[str] = None
    # Remove leftover program outputs before re-measuring a known config, so an
    # interrupted run can be repeated even if the program refuses to overwrite.
    clean_run_outputs: bool = True
    clean_globs: List[str] = field(default_factory=list)


# -----------------------------
# Helpers: path resolution
# -----------------------------
def _results_csv_path(cfg: Config, tabu: TabuSpec) -> Path:
    """
    Stable path of the results CSV.
    """
    explicit = tabu.results_csv or getattr(cfg, "csv_log", None)
    if explicit:
        path = Path(explicit).expanduser()
    else:
        path = Path.cwd() / "tabu_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_workroot(tabu: TabuSpec, workroot: Optional[Union[str, Path]]) -> Path:
    """Directory holding builds and run scratch."""
    if workroot:
        return Path(workroot).expanduser()
    env_root = os.environ.get("SCOUT_TABU_WORKROOT")
    if env_root:
        return Path(env_root).expanduser()
    if tabu.run_dir:
        return Path(tabu.run_dir).expanduser()
    return Path(tempfile.gettempdir()) / "SCOuT_tabu"


# -----------------------------
# Helpers: env enumeration
# -----------------------------
def _enumerate_env_schema(schema: Dict[str, Union[List[str], Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Turn an 'env' schema with optional 'when' predicates into a list of concrete env dicts.
    """
    if not isinstance(schema, dict) or not schema:
        return [{}]
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
            if all(partial.get(k) == str(v) for k, v in pred.items()):
                for v in spec["values"]:
                    partial[var] = str(v)
                    rec(i + 1, partial, out)
                partial.pop(var, None)
            else:
                # predicate not satisfied, skip var entirely
                rec(i + 1, partial, out)
            return

        # Unknown — ignore
        rec(i + 1, partial, out)

    results: List[Dict[str, str]] = []
    rec(0, {}, results)
    # dedup (just in case)
    uniq = {tuple(sorted(d.items())): d for d in results}
    return list(uniq.values())


def _env_combos(cfg: Config, tabu: TabuSpec, rng: random.Random) -> List[Dict[str, str]]:
    """Return a non-empty list of env dicts; fall back to [{}] if cfg.env is empty/missing."""
    mode = (tabu.env_mode or "product").lower()
    if mode == "fixed":
        return [dict(tabu.env or {})]

    # Normal (product/sample) modes
    schema = getattr(cfg, "env", None)
    combos = _enumerate_env_schema(schema if isinstance(schema, dict) else {})
    if not combos:
        combos = [{}]

    cap = tabu.env_cap or 0
    if cap <= 0 or len(combos) <= cap:
        return combos

    if mode == "sample" and len(combos) > 0:
        return rng.sample(combos, cap)

    # default/product: deterministic slice after shuffle for diversity
    rng.shuffle(combos)
    return combos[:cap] if cap > 0 else combos


# -----------------------------
# Flag rendering
# -----------------------------
def _render_flags(
    cfg: Config,
    base_flags: str,
    variant: Optional[str],
    params_choice: Dict[str, Any],
    pool_set: Sequence[str],
) -> Tuple[str, str]:
    """
    Return (flags_key_for_csv, flags_shell_string).
    params_choice: { "-march": "native", "-mllvm -force-vector-interleave": 4, ... }
    """
    parts: List[str] = []
    label_parts: List[str] = []

    if base_flags:
        parts.append(base_flags)

    if variant and variant != base_flags:
        parts.append(variant)
        label_parts.append(variant)

    # params
    for opt, spec in (cfg.compiler_params or {}).items():
        if opt not in params_choice:
            continue
        val = params_choice[opt]
        if "{}" in opt:
            frag = opt.format(val)
        elif isinstance(spec, dict) and "sep" in spec:
            sep = spec.get("sep", "=")
            frag = f"{opt}{sep}{val}"
        else:
            # default "=" glue
            frag = f"{opt}={val}"
        parts.append(frag)
        label_parts.append(frag)

    # pool
    for f in pool_set:
        parts.append(f)
        label_parts.append(f)

    flags_str = " ".join(parts).strip()
    # Optuna-like pretty ID: join the *actual* fragments with |
    # (we prefer using parts, not label_parts, so base flags are present in the key)
    flags_key = "|".join(parts) if parts else "default"
    return flags_key, flags_str


def _active_flag_strings(
    cfg: Config,
    variant: Optional[str],
    params_choice: Dict[str, Any],
) -> List[str]:
    """Exact flag strings a state activates (variant + rendered compiler params)."""
    active: List[str] = []
    if variant:
        active.append(variant)
    for opt, spec in (cfg.compiler_params or {}).items():
        if opt not in params_choice:
            continue
        val = params_choice[opt]
        if "{}" in opt:
            active.append(opt.format(val))
        elif isinstance(spec, dict) and "sep" in spec:
            active.append(f"{opt}{spec.get('sep', '=')}{val}")
        else:
            active.append(f"{opt}={val}")
    return active


def _sanitize_pool(
    cfg: Config,
    variant: Optional[str],
    params_choice: Dict[str, Any],
    pool_list: Sequence[str],
) -> List[str]:
    """Drop pool flags made redundant or conflicting by the variant/params."""
    constraints = getattr(cfg, "flag_constraints", None)
    if constraints is None or constraints.is_empty:
        return list(pool_list)
    active = _active_flag_strings(cfg, variant, params_choice)
    return constraints.sanitize_pool(active, list(pool_list))


# -----------------------------
# Neighborhood moves
def _neighbors(
    cfg: Config,
    state: Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]],
    tabu: TabuSpec,
    rng: random.Random,
    num: int,
    PARAM_MIN: int,
    PARAM_MAX: int,
) -> List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]]:
    variant, params_choice, pool_list, env_dict = state
    pool_all: List[str] = list(cfg.compiler_flag_pool or [])
    variants_all: List[str] = list(cfg.compiler_flags or [])
    params_schema = cfg.compiler_params or {}
    always = set((cfg.compiler_params_select or {}).get("always", []))
    constraints = getattr(cfg, "flag_constraints", None)
    active_now = _active_flag_strings(cfg, variant, params_choice)

    # Helper to get domain of a param
    def _values_for_param(key: str) -> List[Any]:
        spec = params_schema[key]
        return list(spec["values"]) if isinstance(spec, dict) and "values" in spec else list(spec)

    neigh: List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]] = []

    # ---- VARIANT NEIGHBORS (change variant) ----
    if tabu.allow_variant_moves and variants_all:
        cand_variants = [v for v in variants_all if v != variant]
        rng.shuffle(cand_variants)
        for nv in cand_variants:
            # A different variant may make some pool flags redundant/conflicting.
            npool = _sanitize_pool(cfg, nv, params_choice, pool_list)
            neigh.append((nv, dict(params_choice), npool, dict(env_dict)))

    # ---- PARAM NEIGHBORS (add/remove/change/swap) ----
    if tabu.allow_param_moves and params_schema:
        active = set(params_choice.keys())
        all_keys = list(params_schema.keys())
        inactive = [k for k in all_keys if k not in active]

        # ADD (respect max)
        if len(active) < PARAM_MAX and inactive:
            rng.shuffle(inactive)
            for k in inactive:
                vals = _values_for_param(k)
                if not vals: continue
                new_params = dict(params_choice)
                new_params[k] = rng.choice(vals)
                neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

        # REMOVE (respect min)
        if len(active) > PARAM_MIN and active:
            rem_keys = list(active - always)
            rng.shuffle(rem_keys)
            for k in rem_keys:
                new_params = dict(params_choice)
                new_params.pop(k, None)
                neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

        # CHANGE (pick different value)
        if active:
            chg_keys = list(active)
            rng.shuffle(chg_keys)
            for k in chg_keys:
                vals = _values_for_param(k)
                if not vals: continue
                cur = params_choice.get(k)
                choices = [v for v in vals if v != cur] or vals
                if len(choices) == 1 and choices[0] == cur:
                    continue  # no real change available
                new_params = dict(params_choice)
                new_params[k] = rng.choice(choices)
                if new_params[k] == cur:
                    continue
                neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

        # SWAP (remove one, add a different one)
        if active and inactive and (len(active) >= PARAM_MIN) and (len(active) <= PARAM_MAX):
            rem_keys = list(active - always); rng.shuffle(rem_keys)
            add_keys = list(inactive); rng.shuffle(add_keys)
            for rk in rem_keys:
                for ak in add_keys:
                    vals = _values_for_param(ak)
                    if not vals: continue
                    new_params = dict(params_choice)
                    new_params.pop(rk, None)
                    new_params[ak] = rng.choice(vals)
                    neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

    # ---- POOL NEIGHBORS (add/remove single flag) ----
    if tabu.allow_pool_moves and pool_all:
        s = set(pool_list)
        # ADD flags not present (skip those that conflict with or duplicate the
        # current context, or that would make an active flag redundant)
        addables: List[str] = []
        for f in pool_all:
            if f in s:
                continue
            if constraints is not None and not constraints.is_empty:
                if not constraints.can_add(f, active_now + list(pool_list)):
                    continue
            addables.append(f)
        rng.shuffle(addables)
        for f in addables:
            new_pool = sorted(s | {f})
            neigh.append((variant, dict(params_choice), new_pool, dict(env_dict)))
        # REMOVE flags present
        removables = list(s)
        rng.shuffle(removables)
        for f in removables:
            new_pool = sorted(s - {f})
            neigh.append((variant, dict(params_choice), new_pool, dict(env_dict)))

    # ---- DEDUP + NO-OP FILTER + CAP ----
    uniq: List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]] = []
    seen = set()
    cur_key = (
        variant,
        tuple(sorted(params_choice.items())),
        tuple(pool_list),
        tuple(sorted(env_dict.items())),
    )
    rng.shuffle(neigh)  # randomize before slicing

    for nv, nparams, npool, nenv in neigh:
        if constraints is not None and not constraints.is_empty:
            if not constraints.is_allowed(_active_flag_strings(cfg, nv, nparams) + list(npool)):
                continue
        key = (
            nv,
            tuple(sorted(nparams.items())),
            tuple(npool),
            tuple(sorted(nenv.items())),
        )
        if key == cur_key:  # no-op, skip
            continue
        if key in seen:     # duplicate, skip
            continue
        seen.add(key)
        uniq.append((nv, nparams, npool, nenv))
        if len(uniq) >= max(1, num):   # respect requested neighborhood size
            break

    return uniq



# -----------------------------
# Main entry
# -----------------------------
def run_tabu_study(
    cfg: Config,
    *,
    resume: bool = False,
    iters: Optional[int] = None,
    budget: str = "lifetime",
    workroot: Optional[Union[str, Path]] = None,
) -> None:
    """
    Tabu search over compiler flags and runtime environments.

    Every measurement is appended to the results CSV immediately and each
    completed iteration snapshots the search state, so an interrupted run
    (for example, walltime kill) can be continued with ``resume=True``.

    Configurations that already have a recorded result are not built or measured
    again.
    
    Builds are only reused when their artifacts are still on disk.
    """
    budget = str(budget or "lifetime").lower()
    if budget not in {"lifetime", "per-run"}:
        raise ValueError("budget must be 'lifetime' or 'per-run'")

    # TabuSpec from cfg.tabu (if provided) or defaults
    raw = getattr(cfg, "tabu", {}) or {}
    tabu = TabuSpec(**{k: v for k, v in raw.items() if k in TabuSpec.__annotations__})

    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))

    objective = cfg.objectives[0]
    metric_name = objective.metric
    goal = objective.goal
    goal_min = (goal == "min")

    def score(v: float) -> float:
        """Lower is always better internally."""
        return v if goal_min else -v

    base = cfg.compiler_flags_base or ""
    variants = list(cfg.compiler_flags or [])
    params_schema = cfg.compiler_params or {}

    sel = getattr(cfg, "compiler_params_select", {}) or {}
    if "k" in sel:
        PARAM_MIN = PARAM_MAX = int(sel["k"])
    else:
        PARAM_MIN = int(sel.get("min", 0))
        PARAM_MAX = int(sel.get("max", len(params_schema)))
    always = [k for k in sel.get("always", []) if k in params_schema]
    PARAM_MIN = max(PARAM_MIN, len(always))
    if PARAM_MAX < PARAM_MIN:
        PARAM_MAX = PARAM_MIN

    # ------------------------------------------------------------------
    # Durable journal: results CSV + failure CSV + search state snapshot
    # ------------------------------------------------------------------
    fingerprint = config_fingerprint(cfg, tabu)
    results_csv = _results_csv_path(cfg, tabu)
    journal = RunJournal(
        results_csv,
        objective_metrics=[metric_name],
        state_path=tabu.state_file,
        failure_path=tabu.failure_csv or getattr(cfg, "fail_log", None),
    )
    previous = journal.load()

    state: Dict[str, Any] = {}
    if resume:
        if previous is None:
            print("[tabu] --resume: no checkpoint found; recording into the configured files")
        else:
            stored = str(previous.get("fingerprint") or "")
            if stored and stored != fingerprint:
                raise SystemExit(
                    "[tabu] refusing to resume: the configuration no longer matches the checkpoint.\n"
                    f"       checkpoint: {stored}\n"
                    f"       current:    {fingerprint}\n"
                    "       Re-run without --resume to start a new search."
                )
            state = dict(previous.get("search") or {})
    else:
        archived = journal.archive()
        previous = None
        if archived is not None:
            print(f"[tabu] previous run archived: {archived}")

    known_records = journal.known_records()
    known_failures = journal.known_failures()

    workroot = _resolve_workroot(tabu, workroot)
    workroot.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(cfg, workroot)

    restored = 0
    for record in known_records.values():
        if record.flags and evaluator.seed_evaluation(
            record.flags, record.env, record.metrics, record.binary
        ):
            restored += 1
    for entry in known_failures.values():
        flags = str(entry.get("flags") or "")
        if flags:
            evaluator.seed_failure(flags, str(entry.get("reason") or "previous failure"))

    print(f"[tabu] workdir: {workroot}")
    print(f"[tabu] results CSV: {results_csv}")
    if known_records or known_failures:
        print(
            f"[tabu] journal: {len(known_records)} measured + {len(known_failures)} failed "
            f"config(s) on record ({restored} result(s) restored)"
        )

    # ------------------------------------------------------------------
    # Environments, RNG and the state to continue from
    # ------------------------------------------------------------------
    restored_envs = [dict(e) for e in (state.get("env_list") or [])]
    if restored_envs:
        env_list = restored_envs
    else:
        env_list = _env_combos(cfg, tabu, rng)
    if not env_list:
        env_list = [{}]
    print(f"[tabu] env_mode={tabu.env_mode} env_combos={len(env_list)}")

    # Restore RNG state if present in the checkpoint
    stored_rng = state.get("rng_state")
    if stored_rng:
        try:
            rng.setstate(rng_from_json(stored_rng))
        except (TypeError, ValueError) as exc:
            print(f"[tabu] warning: could not restore the RNG state ({exc}); using the config seed")

    started = bool(state)
    current = state.get("current") or {}
    if current:
        variant: Optional[str] = current.get("variant")
        params_choice: Dict[str, Any] = dict(current.get("params") or {})
        pool_list: List[str] = list(current.get("pool") or [])
        env: Dict[str, str] = dict(current.get("env") or {})
    else:
        # Initial state: start small (base + first variant; params unset; pool empty)
        variant = variants[0] if variants else None
        params_choice = {}
        pool_list = []
        if PARAM_MIN > 0 and params_schema:
            keys = always + [k for k in params_schema if k not in always]
            rng.shuffle(keys)
            if always:
                keys = always + [k for k in keys if k not in always]
            need = min(PARAM_MIN, len(keys))
            for k in keys[:need]:
                spec = params_schema[k]
                if isinstance(spec, dict) and "values" in spec:
                    vals = list(spec["values"])
                else:
                    vals = list(spec)
                if not vals:
                    continue
                params_choice[k] = rng.choice(vals)
        env = dict(env_list[0]) if env_list else {}

    # Restored states may predate a constraint change: drop anything now
    # redundant or conflicting before the baseline is evaluated.
    pool_list = _sanitize_pool(cfg, variant, params_choice, pool_list)

    phase = str(state.get("phase") or ("search" if started else "baseline"))
    iters_done = int(state.get("iteration", 0) or 0)
    no_improve = int(state.get("no_improve", 0) or 0)
    if not started and known_records:
        # The results CSV survived but the snapshot did not: keep the recorded
        # history (and the iteration numbering) instead of restarting from zero.
        iters_done = max(int(record.k) for record in known_records.values())
        print(
            f"[tabu] no checkpoint found: reusing {len(known_records)} recorded result(s) "
            f"and continuing from iteration {iters_done}"
        )

    # Tabu memory: store config keys; aspiration allows override if improves best
    tabu_q: deque[str] = deque(state.get("tabu_queue") or [], maxlen=tabu.tabu_tenure)

    best_val = math.inf
    best_key = ""
    best_flags = ""
    best_env: Dict[str, str] = {}
    best_metrics: Dict[str, float] = {}
    best_binary = ""
    stored_best = state.get("best") or {}
    if stored_best:
        best_val = float(stored_best.get("value", math.inf))
        best_key = str(stored_best.get("key") or "")
        best_flags = str(stored_best.get("flags") or "")
        best_env = dict(stored_best.get("env") or {})
        best_metrics = {str(k): float(v) for k, v in (stored_best.get("metrics") or {}).items()}
        best_binary = str(stored_best.get("binary") or "")

    if started and phase == "search":
        print(
            f"[tabu] resuming: {iters_done} iteration(s) done, "
            f"best {metric_name}={best_val:.6g}, no_improve={no_improve}"
        )

    def snapshot(phase_name: str) -> None:
        """Atomically persist where the search is, so a restart can continue."""
        journal.save_state(
            fingerprint=fingerprint,
            search={
                "iteration": iters_done,
                "no_improve": no_improve,
                "phase": phase_name,
                "env_list": env_list,
                "rng_state": rng_to_json(rng.getstate()),
                "tabu_queue": list(tabu_q),
                "current": {
                    "variant": variant,
                    "params": params_choice,
                    "pool": pool_list,
                    "env": dict(env),
                },
                # Do we have a best result yet?
                "best": None
                if math.isinf(best_val)
                else {
                    "value": best_val,
                    "key": best_key,
                    "flags": best_flags,
                    "env": best_env,
                    "metrics": best_metrics,
                    "binary": best_binary,
                },
            },
            meta={"results_csv": str(results_csv)},
        )

    def evaluate_one(k: int, flags_str: str, flags_key: str, env_dict: Dict[str, str]):
        """
        Return (value, metrics, binary, eval_id) for one configuration.

        Recorded results are reused as-is; anything not on record is built and
        measured, then appended to the CSV before the next step starts.
        """
        compile_key = evaluator.compile_key(flags_str)
        eid = eval_id(compile_key, env_dict)
        record = journal.lookup(eid)
        if record is not None:
            return record.value, dict(record.metrics), record.binary, eid

        reason = journal.failure_reason(eid)
        if reason is not None:
            raise RuntimeError(f"known failure: {reason}")

        run_dir = evaluator.run_dir_for(flags_str, env_dict)
        if tabu.clean_run_outputs:
            clean_run_outputs(run_dir, cfg.program_args, tabu.clean_globs)
        try:
            evaluation = evaluator.evaluate(flags_str, env_dict, run_dir)
            if metric_name not in evaluation.metrics:
                raise RuntimeError(
                    f"objective metric '{metric_name}' missing; got {list(evaluation.metrics)}"
                )
        except Exception as ex:
            journal.record_failure(
                k=k, eid=eid, flags=flags_str, env=env_dict, reason=str(ex)
            )
            raise

        metrics = dict(evaluation.metrics)
        record = EvalRecord(
            k=k,
            eval_id=eid,
            value=float(metrics[metric_name]),
            flags_key=flags_key,
            flags=flags_str,
            env=dict(env_dict),
            binary=str(evaluation.binary),
            compile_key=compile_key,
            metrics=metrics,
            cached=evaluation.cached,
        )
        journal.append(record)
        return record.value, metrics, record.binary, eid


    # ------------------------------------------------------------------
    # Baseline: pick the best environment for the starting flags
    # ------------------------------------------------------------------
    best_start: Optional[Tuple[str, str, Dict[str, str], float, MetricDict, str]] = None
    best_start_sc = math.inf
    if phase == "baseline":
        print("[tabu] evaluating baseline across environments")
        for i, e in enumerate(env_list, 1):
            key, flags_str = _render_flags(cfg, base, variant, params_choice, pool_list)
            try:
                v, mets, binp, _eid = evaluate_one(0, flags_str, key, e)
                sc = score(v)
            except Exception as ex:
                print(f"[tabu] baseline env{i:03d} FAILED: {ex}")
                continue
            if sc < best_start_sc:
                best_start_sc = sc
                best_start = (key, flags_str, dict(e), v, mets, binp)

        if best_start is None:
            raise RuntimeError("tabu: all baseline evaluations failed.")

        key, flags_str, env_best, v, mets, binp = best_start
        env = dict(env_best)
        best_val = v
        best_key = key
        best_flags = flags_str
        best_env = dict(env_best)
        best_metrics = dict(mets)
        best_binary = binp
        no_improve = 0
        tabu_q.clear()
        tabu_q.append(key + "|" + json.dumps(env, sort_keys=True))
        phase = "search"
        snapshot(phase)
        print(f"[tabu] start: {metric_name}={best_val:.6g} env={json.dumps(best_env)}")

    # ------------------------------------------------------------------
    # Search loop
    # ------------------------------------------------------------------
    if budget == "per-run":
        stop_after = iters_done + int(iters if iters is not None else tabu.max_iters)
    else:
        stop_after = int(iters if iters is not None else tabu.max_iters)
    print(f"[tabu] iterations done: {iters_done}; this run stops at {stop_after} ({budget} budget)")

    stop_reason = "iteration budget reached"
    finished = True
    install_sigterm_handler()
    try:
        while iters_done < stop_after and no_improve < tabu.max_no_improve:
            attempt = iters_done + 1
            print(
                f"[tabu] ===== iter {attempt}/{stop_after} "
                f"(best={best_val:.6g}, no_improve={no_improve}) ====="
            )

            # Generate neighbors for the current state (flag-side)
            neigh = _neighbors(cfg, (variant, params_choice, pool_list, env), tabu, rng, tabu.neighborhood, PARAM_MIN, PARAM_MAX,)

            # Add environment neighbors if allowed
            if tabu.allow_env_moves and len(env_list) > 1:
                cur_env_key = tuple(sorted(env.items()))
                env_unique = [e for e in env_list if tuple(sorted(e.items())) != cur_env_key]
                k = min(8, len(env_unique))
                env_additions = [(variant, dict(params_choice), list(pool_list), dict(e))
                                 for e in (rng.sample(env_unique, k) if k > 0 else [])]
                # dedup against already-built neighbors
                seen_keys = {
                    (nv, tuple(sorted(np.items())), tuple(npl), tuple(sorted(ne.items())))
                    for nv, np, npl, ne in neigh
                }
                for cand in env_additions:
                    key = (cand[0], tuple(sorted(cand[1].items())), tuple(cand[2]), tuple(sorted(cand[3].items())))
                    if key not in seen_keys:
                        neigh.append(cand)
                        seen_keys.add(key)

            # Evaluate neighbors, pick the best admissible (or aspirated)
            cand_best: Optional[Tuple[Any, ...]] = None
            cand_best_sc = math.inf

            for nv, nparams, npool, nenv in neigh:
                nkey, nflags = _render_flags(cfg, base, nv, nparams, npool)
                # admissible if not tabu OR aspirates (improves global best)
                is_tabu = (nkey + "|" + json.dumps(nenv, sort_keys=True)) in tabu_q

                try:
                    v, mets, binp, _eid = evaluate_one(attempt, nflags, nkey, nenv)
                    sc = score(v)
                except Exception as ex:
                    message = str(ex)
                    if not message.startswith("known failure"):
                        print(f"[tabu] iter {attempt}: candidate failed, {message}")
                    continue

                aspirates = sc < score(best_val)
                if is_tabu and not aspirates:
                    continue
                if sc < cand_best_sc:
                    cand_best_sc = sc
                    cand_best = (nv, nparams, npool, nenv, v, mets, binp, nkey, nflags)

            if cand_best is None:
                print(f"[tabu] iter {attempt}: no admissible neighbor; stopping.")
                stop_reason = "no admissible neighbor"
                finished = False
                break

            # Move to the best candidate and record the move as tabu
            variant, params_choice, pool_list, env = cand_best[0], cand_best[1], cand_best[2], cand_best[3]
            v, mets, binp, nkey, nflags = cand_best[4], cand_best[5], cand_best[6], cand_best[7], cand_best[8]
            tabu_q.append(nkey + "|" + json.dumps(env, sort_keys=True))
            iters_done = attempt

            improved = score(v) < score(best_val)
            meaningful = is_significant_improvement(
                best_val, v, goal,
                cfg.significance.min_rel_gain, cfg.significance.min_abs_gain,
            )
            if improved:
                best_val = v
                best_key = nkey
                best_flags = nflags
                best_env = dict(env)
                best_metrics = dict(mets)
                best_binary = binp
                print(f"[tabu] iter {attempt}: IMPROVED: {metric_name}={best_val:.6g}")
            if meaningful:
                no_improve = 0
            else:
                no_improve += 1
                if not improved:
                    print(f"[tabu] iter {attempt}: best={best_val:.6g} (no_improve={no_improve})")

            # One completed iteration is the checkpoint granularity: the
            # evaluations made inside it are already in the CSV.
            snapshot("search")

        if finished and no_improve >= tabu.max_no_improve:
            stop_reason = "no-improvement limit reached"
    except (KeyboardInterrupt, StopRequested) as exc:
        stop_reason = f"interrupted ({type(exc).__name__})"
        finished = False
        print(f"\n[tabu] {stop_reason}; saving checkpoint…")
    finally:
        restore_sigterm_handler()
        snapshot(phase)

    print("\n[tabu] ===== Summary =====")
    if math.isinf(best_val):
        print("best: <no successful measurement>")
    else:
        print(f"best: {metric_name}={best_val:.6g}")
        print(f"flags: {best_key}")
        print(f"env: {json.dumps(best_env)}")
    if iters_done > stop_after:
        print(
            f"[tabu] iterations completed: {iters_done} "
            f"(the budget of {stop_after} was already used up; {stop_reason})"
        )
    else:
        print(f"[tabu] iterations completed: {iters_done}/{stop_after} ({stop_reason})")
    print(f"[tabu] results: {results_csv}")
    print(f"[tabu] state: {journal.state_path}")
    if not finished:
        print("[tabu] resume with the same command plus --resume")
    print(f"[tabu] unique builds: {evaluator.build_count}")
