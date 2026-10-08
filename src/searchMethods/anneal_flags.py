from __future__ import annotations

###############################################################################
# Standard library imports
###############################################################################
import csv, json, math, random, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

###############################################################################
# Local imports
###############################################################################
from src.config import Config
from src.evaluator import Evaluator
from src.misc import unique_csv_path, is_significant_improvement

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
MetricDict = Dict[str, Number]

# ---------- Objective / scoring ----------
def _choose_objective(cfg: Config) -> Tuple[str, str]:
    if not cfg.objectives:
        raise ValueError("No objectives in config.")
    obj = cfg.objectives[0]
    return obj.metric, ("min" if obj.goal == "min" else "max")

def _score(value: float, goal: str) -> float:
    return value if goal == "min" else -value

# ---------- Env enumeration (same semantics you use elsewhere) ----------
def _enumerate_env_schema(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    keys = list(schema.keys())
    out: List[Dict[str, str]] = []
    def rec(i: int, partial: Dict[str, str]):
        if i == len(keys):
            out.append(dict(partial)); return
        var = keys[i]; spec = schema[var]
        if isinstance(spec, list):
            for v in spec:
                partial[var] = str(v); rec(i+1, partial)
            partial.pop(var, None); return
        if isinstance(spec, dict) and "values" in spec:
            pred = spec.get("when", {})
            if all(partial.get(k) == str(v) for k, v in pred.items()):
                for v in spec["values"]:
                    partial[var] = str(v); rec(i+1, partial)
                partial.pop(var, None)
            else:
                rec(i+1, partial)
            return
        rec(i+1, partial)
    rec(0, {})
    uniq = {tuple(sorted(d.items())): d for d in out}
    return list(uniq.values())

def _env_combos(cfg: Config, mode: str, cap: Optional[int], rng: random.Random, fixed: Optional[Dict[str,str]]=None) -> List[Dict[str,str]]:
    mode = (mode or "product").lower()
    if mode == "fixed":
        return [dict(fixed or {})]
    schema = getattr(cfg, "env", {}) or {}
    combos = _enumerate_env_schema(schema)
    if not combos: return [{}]
    if not cap or cap <= 0 or len(combos) <= cap: return combos
    if mode == "sample":
        return rng.sample(combos, cap)
    rng.shuffle(combos)  # product: shuffle then slice for diversity
    return combos[:cap]

# ---------- Render flags (copied from tabu style) ----------
def _render_flags(
    cfg: Config,
    base_flags: str,
    variant: Optional[str],
    params_choice: Dict[str, Any],
    pool_set: Sequence[str],
) -> Tuple[str, str]:
    parts: List[str] = []
    if base_flags:
        parts.append(base_flags)
    if variant and variant != base_flags:
        parts.append(variant)
    for opt, spec in (getattr(cfg, "compiler_params", {}) or {}).items():
        if opt not in params_choice:
            continue
        val = params_choice[opt]
        if "{}" in opt:
            frag = opt.format(val)
        elif isinstance(spec, dict) and "sep" in spec:
            frag = f"{opt}{spec.get('sep', '=')}{val}"
        else:
            frag = f"{opt}={val}"
        parts.append(frag)
    for f in pool_set:
        parts.append(f)
    flags_str = " ".join(parts).strip()
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
    for opt, spec in (getattr(cfg, "compiler_params", {}) or {}).items():
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


# ---------- Build & measure ----------
def _compile_and_measure(evaluator: Evaluator, cfg: Config, flags_str: str, env: Dict[str, str], work: Path) -> Tuple[float, MetricDict, str]:
    evaluation = evaluator.evaluate(flags_str, env, work)
    metrics = evaluation.metrics
    binary = evaluation.binary
    metric_name, _goal = _choose_objective(cfg)
    if metric_name not in metrics:
        raise RuntimeError(f"objective metric '{metric_name}' missing; got {list(metrics.keys())}")
    return float(metrics[metric_name]), metrics, str(binary)

# ---------- Anneal params ----------
@dataclass
class _AnnealParams:
    T0: float = 1.0
    alpha: float = 0.95
    max_iters: int = 200
    max_no_improve: int = 30
    neighbor_mode: str = "mix"
    env_mode: str = "product"
    env_cap: Optional[int] = 8
    env: Dict[str, str] = field(default_factory=dict)
    results_csv: Optional[str] = None

def _params_from_cfg(cfg: Config) -> _AnnealParams:
    p = _AnnealParams()

    def _coerce(dst: _AnnealParams, block: dict) -> None:
        # Only overwrite if present in block
        if "T0"             in block and block["T0"] is not None:            dst.T0 = float(block["T0"])
        if "alpha"          in block and block["alpha"] is not None:         dst.alpha = float(block["alpha"])
        if "max_iters"      in block and block["max_iters"] is not None:     dst.max_iters = int(block["max_iters"])
        if "max_no_improve" in block and block["max_no_improve"] is not None:dst.max_no_improve = int(block["max_no_improve"])
        if "neighbor_mode"  in block and block["neighbor_mode"] is not None: dst.neighbor_mode = str(block["neighbor_mode"])
        if "env_mode"       in block and block["env_mode"] is not None:      dst.env_mode = str(block["env_mode"])
        if "env_cap" in block:  # allow None
            dst.env_cap = None if block["env_cap"] is None else int(block["env_cap"])
        if "env"            in block and block["env"] is not None:          dst.env = dict(block["env"])
        if "results_csv"    in block:                                        dst.results_csv = block["results_csv"]

    # 1) Strong preference: top-level cfg.anneal (attribute or mapping)
    block = getattr(cfg, "anneal", None)
    if isinstance(block, dict):
        _coerce(p, block); return p
    if block is not None and not isinstance(block, dict):
        # object-like with attributes
        for k in p.__dataclass_fields__.keys():
            if hasattr(block, k):
                setattr(p, k, getattr(block, k))
        return p

    # 2) Raw dicts some Configs keep around
    for attr in ("_raw", "_data", "raw", "config", "__dict__"):
        raw = getattr(cfg, attr, None)
        if isinstance(raw, dict) and "anneal" in raw and isinstance(raw["anneal"], dict):
            _coerce(p, raw["anneal"]); return p

    # 3) Optional: allow cfg.search.anneal ONLY if you actually put params there
    s = getattr(cfg, "search", None)
    if isinstance(s, dict) and isinstance(s.get("anneal"), dict):
        _coerce(p, s["anneal"]); return p
    if s is not None and hasattr(s, "anneal") and isinstance(getattr(s, "anneal"), dict):
        _coerce(p, getattr(s, "anneal")); return p

    # 4) Nothing found → keep defaults
    return p



# ---------- Neighbor generator (variant / params / pool), with min/max bounds ----------
def _neighbors_sa(
    cfg: Config,
    state: Tuple[Optional[str], Dict[str, Any], List[str]],
    rng: random.Random,
    param_min: int,
    param_max: int,
    num: int = 24,
) -> List[Tuple[Optional[str], Dict[str, Any], List[str]]]:
    variant, params_choice, pool_list = state
    variants_all: List[str] = list(cfg.compiler_flags or [])
    pool_all: List[str]     = list(cfg.compiler_flag_pool or [])
    params_schema           = cfg.compiler_params or {}
    always = set((cfg.compiler_params_select or {}).get("always", []))
    constraints = getattr(cfg, "flag_constraints", None)
    active_variant_params = _active_flag_strings(cfg, variant, params_choice)

    out: List[Tuple[Optional[str], Dict[str, Any], List[str]]] = []

    def values_for_param(key: str) -> List[Any]:
        spec = params_schema[key]
        return list(spec["values"]) if isinstance(spec, dict) and "values" in spec else list(spec)

    def move_variant():
        choices = [v for v in variants_all if v != variant]
        if not choices: return
        nv = rng.choice(choices)
        new_pool = list(pool_list)
        if constraints is not None and not constraints.is_empty:
            new_pool = constraints.sanitize_pool(
                _active_flag_strings(cfg, nv, params_choice), pool_list
            )
        out.append((nv, dict(params_choice), new_pool))

    def move_param():
        if not params_schema: return
        active = set(params_choice.keys()); all_keys = list(params_schema.keys())
        rng.shuffle(all_keys)

        actions = []
        if len(active) < param_max: actions.append("add")
        if len(active) > param_min: actions.append("remove")
        if len(active) > 0:         actions.append("change")
        if len(active) > 0 and len(active) < len(all_keys): actions.append("swap")
        if not actions:
            if len(active) > 0: actions = ["change"]
            else: return
        act = rng.choice(actions)

        if act == "add":
            candidates = [k for k in all_keys if k not in active]
            if not candidates: return
            k = rng.choice(candidates); vals = values_for_param(k)
            if not vals: return
            new_params = dict(params_choice); new_params[k] = rng.choice(vals)
            out.append((variant, new_params, list(pool_list)))
        elif act == "remove":
            if not active: return
            removable = list(active - always)
            if not removable: return
            k = rng.choice(removable)
            new_params = dict(params_choice); new_params.pop(k, None)
            out.append((variant, new_params, list(pool_list)))
        elif act == "change":
            if not active: return
            k = rng.choice(list(active)); vals = values_for_param(k)
            if not vals: return
            cur = params_choice.get(k); choices = [v for v in vals if v != cur]
            if not choices: return
            new_params = dict(params_choice); new_params[k] = rng.choice(choices)
            out.append((variant, new_params, list(pool_list)))
        elif act == "swap":
            if not active or len(active) == len(all_keys): return
            removable = list(active - always)
            if not removable: return
            rem_k = rng.choice(removable)
            add_candidates = [k for k in all_keys if k not in active]
            if not add_candidates: return
            add_k = rng.choice(add_candidates); vals = values_for_param(add_k)
            if not vals: return
            new_params = dict(params_choice); new_params.pop(rem_k, None); new_params[add_k] = rng.choice(vals)
            out.append((variant, new_params, list(pool_list)))

    def move_pool():
        if not pool_all: return
        new_pool = set(pool_list)
        if rng.random() < 0.5 and new_pool:
            rem = rng.choice(list(new_pool)); new_pool.remove(rem)
        else:
            cand = rng.choice(pool_all)
            if cand in new_pool:
                new_pool.remove(cand)
            elif constraints is None or constraints.is_empty or constraints.can_add(
                cand, active_variant_params + list(pool_list)
            ):
                new_pool.add(cand)
            elif new_pool:
                # Adding `cand` is illegal: fall back to removing one instead.
                new_pool.remove(rng.choice(list(new_pool)))
        out.append((variant, dict(params_choice), sorted(new_pool)))

    moves = []
    if variants_all:   moves.append(move_variant)
    if params_schema:  moves.append(move_param)
    if pool_all:       moves.append(move_pool)
    if not moves: return out

    for _ in range(num):
        rng.choice(moves)()
    current_key = (variant, tuple(sorted(params_choice.items())), tuple(sorted(pool_list)))
    unique = {}
    for candidate in out:
        key = (candidate[0], tuple(sorted(candidate[1].items())), tuple(sorted(candidate[2])))
        if key != current_key:
            unique[key] = candidate
    return list(unique.values())

# ---------- Main SA study ----------
def run_anneal_study(cfg: Config) -> None:
    params = _params_from_cfg(cfg)
    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))

    # Base + variants + params + pool, same init as tabu
    base_flags: str = cfg.compiler_flags_base or ""
    variants = list(cfg.compiler_flags or [])
    variant0: Optional[str] = variants[0] if variants else None
    params_choice: Dict[str, Any] = {}
    pool_list: List[str] = []

    # Respect compiler_params_select bounds for the number of active params
    sel = getattr(cfg, "compiler_params_select", {}) or {}
    if "k" in sel:
        PARAM_MIN = PARAM_MAX = int(sel["k"])
    else:
        PARAM_MIN = int(sel.get("min", 0))
        PARAM_MAX = int(sel.get("max", len((cfg.compiler_params or {}))))
    always = [k for k in sel.get("always", []) if k in (cfg.compiler_params or {})]
    PARAM_MIN = max(PARAM_MIN, len(always))
    if PARAM_MAX < PARAM_MIN:
        PARAM_MAX = PARAM_MIN
    # seed with PARAM_MIN random params (if any)
    params_schema = cfg.compiler_params or {}
    if PARAM_MIN > 0 and params_schema:
        keys = [k for k in params_schema if k not in always]; rng.shuffle(keys)
        keys = always + keys
        for k in keys[:min(PARAM_MIN, len(keys))]:
            spec = params_schema[k]
            vals = list(spec["values"]) if isinstance(spec, dict) and "values" in spec else list(spec)
            if vals: params_choice[k] = rng.choice(vals)

    # Objective / significance
    metric_name, goal = _choose_objective(cfg)
    sig = cfg.significance
    MIN_REL = sig.min_rel_gain
    MIN_ABS = sig.min_abs_gain

    # Environments
    envs = _env_combos(cfg, params.env_mode, params.env_cap, rng, fixed=params.env)
    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_anneal_"))
    evaluator = Evaluator(cfg, workroot)
    print(f"[anneal] workdir={workroot} T0={params.T0} alpha={params.alpha} iters={params.max_iters} envs={len(envs)}")

    # Evaluate a state (variant, params, pool) across envs; return best env outcome
    def eval_state(variant: Optional[str], params_choice: Dict[str, Any], pool_list: List[str]
                   ) -> Tuple[float, Dict[str,float], str, Dict[str,str], str]:
        best_score = math.inf
        best_pkg = None
        key, flags_str = _render_flags(cfg, base_flags, variant, params_choice, pool_list)
        for i, env in enumerate(envs, 1):
            run_dir = workroot / "eval" / f"env{i:03d}"
            try:
                v, m, b = _compile_and_measure(evaluator, cfg, flags_str, env, run_dir)
            except Exception as exc:
                continue
            candidate_score = _score(v, goal)
            if candidate_score < best_score:
                best_score = candidate_score
                best_pkg = (v, m, b, env, key)
        if best_pkg is None:
            raise RuntimeError(f"all environment evaluations failed for flags '{key}'")
        return best_pkg

    # Baseline
    cur_variant, cur_params, cur_pool = variant0, dict(params_choice), list(pool_list)
    cur_val, cur_mets, cur_bin, cur_env, cur_key = eval_state(cur_variant, cur_params, cur_pool)
    best_val, best_variant, best_params, best_pool, best_env, best_mets, best_bin, best_key = (
        cur_val, cur_variant, dict(cur_params), list(cur_pool), dict(cur_env), dict(cur_mets), cur_bin, cur_key
    )

    # CSV buffer: Optuna-like schema (no k column requested here; add if you want)
    rows: List[Tuple[List[float], str, Dict[str, str], str, Dict[str, float]]] = []
    extra_metric_keys: set[str] = set(cur_mets.keys())
    rows.append(([float(cur_val)], cur_key, dict(cur_env), str(cur_bin), cur_mets))

    # SA loop
    T = float(max(params.T0, 1e-12))
    no_improve = 0
    for it in range(1, int(params.max_iters) + 1):
        # propose neighbor state
        neighs = _neighbors_sa(cfg, (cur_variant, cur_params, cur_pool), rng, PARAM_MIN, PARAM_MAX, num=24)
        if not neighs:
            print("[anneal] no neighbors; stopping."); break
        rng.shuffle(neighs)
        cand_variant, cand_params, cand_pool = neighs[0]

        cand_val, cand_mets, cand_bin, cand_env, cand_key = eval_state(cand_variant, cand_params, cand_pool)

        # log
        extra_metric_keys.update(cand_mets.keys())
        rows.append(([float(cand_val)], cand_key, dict(cand_env), str(cand_bin), cand_mets))

        # acceptance
        cur_sc = _score(cur_val, goal); new_sc = _score(cand_val, goal)
        accept = (new_sc < cur_sc) or (rng.random() < math.exp(-(new_sc - cur_sc) / max(T, 1e-12)))
        if accept:
            cur_variant, cur_params, cur_pool = cand_variant, cand_params, cand_pool
            cur_val, cur_mets, cur_bin, cur_env, cur_key = cand_val, cand_mets, cand_bin, cand_env, cand_key

        # Always retain the true best. Significance only controls plateau reset.
        improved = _score(cur_val, goal) < _score(best_val, goal)
        meaningful = is_significant_improvement(best_val, cur_val, goal, MIN_REL, MIN_ABS)
        if improved:
            best_val, best_variant, best_params, best_pool, best_env, best_mets, best_bin, best_key = (
                cur_val, cur_variant, dict(cur_params), list(cur_pool), dict(cur_env), dict(cur_mets), cur_bin, cur_key
            )
        if meaningful:
            no_improve = 0
        else:
            no_improve += 1

        # cool + early stop
        T *= float(params.alpha)
        if no_improve >= int(params.max_no_improve):
            print("[anneal] early stop: no-improvement plateau.")
            break

        print(f"[anneal] iter={it} T={T:.4g} cur={cur_val:.6g} best={best_val:.6g} flags={best_key}")

    # write CSV (Optuna-like)
    if getattr(cfg, "csv_log", None) and not params.results_csv:
        results_path = unique_csv_path(cfg.csv_log); Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        results_path = Path(params.results_csv) if params.results_csv else (workroot / "anneal_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    obj_headers = [o.metric for o in cfg.objectives]
    extra_cols = sorted(extra_metric_keys - set(obj_headers))
    header = obj_headers + ["compiler_flags", "env", "binary"] + extra_cols

    with open(results_path, "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(header)
        for obj_vals, flags_key, env_row, binary_path, metrics in rows:
            if not obj_vals or math.isinf(obj_vals[0]): continue
            row = list(obj_vals) + [flags_key, json.dumps(env_row), binary_path]
            row += [metrics.get(k, "") for k in extra_cols]
            w.writerow(row)

    print("\n[anneal] ===== Summary =====")
    print(f"best: {metric_name}={best_val:.6g}")
    print(f"flags: {best_key}")
    print(f"env:   {json.dumps(best_env)}")
    print(f"[anneal] results → {results_path}")
    print(f"[anneal] unique builds: {evaluator.build_count}")
