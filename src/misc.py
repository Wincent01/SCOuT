
###############################################################################
# Standard library imports                                                    #
###############################################################################

import itertools
import shlex
import sys
import tempfile
import hashlib, json, os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
import shutil

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Misc helpers                                                                #
###############################################################################

##### Optuna helpers

# --- complexity schedule based on trial number ---
def _complexity_limits(
    trial,
    n_startup_trials: int,
    n_params_total: int,
    n_pool_total: int,
) -> tuple[int, int]:
    """Return (param_cap, pool_cap) for this trial number.
    Early trials: keep very small; later: gradually relax."""
    t = int(getattr(trial, "number", 0))
    if t < n_startup_trials:                 # stage 0
        return min(1, n_params_total), min(2, n_pool_total)
    elif t < 2 * n_startup_trials:           # stage 1
        return min(2, n_params_total), min(8, n_pool_total)
    else:                                    # stage 2+
        return n_params_total, n_pool_total

# --- choose up to pool_cap on/off flags using scores ---
def _select_pool_flags(trial, pool_flags: list[str], pool_cap: int,
                       constraints=None, active=()) -> list[str]:
    """Pick 0..pool_cap booleans from the pool using scores + a sampled k.

    When ``constraints`` is given, picks that are redundant or mutually
    exclusive are discarded before the flag string is rendered.
    """
    if not pool_flags or pool_cap <= 0:
        return []
    k = int(trial.suggest_int("pool_k", 0, min(pool_cap, len(pool_flags))))
    scored = [(float(trial.suggest_float(f"score::pool::{f}", 0.0, 1.0)), f) for f in pool_flags]
    scored.sort(reverse=True)
    selected = [f for _, f in scored[:k]]
    if constraints is not None and not constraints.is_empty:
        selected = constraints.select_consistent(selected, active)
    selected_set = set(selected)
    return [f for f in pool_flags if f in selected_set]



def _render_one_param(trial, opt: str, spec: Any) -> Tuple[str, str]:
    """
    Returns (key_for_logging, flag_string) for a single compiler param.
    Supports:
      - spec = [v1, v2, ...]  ➜ default "=" glue
      - spec = {"sep": " ", "values": [...]}
      - opt contains "{}" and spec is [values] ➜ format string
    """
    if "{}" in opt and isinstance(spec, list):
        val = trial.suggest_categorical(opt, spec)
        rendered = opt.format(val)
        return rendered, rendered

    if isinstance(spec, list):
        val = trial.suggest_categorical(opt, spec)
        return f"{opt}={val}", f"{opt}={val}"

    if isinstance(spec, dict) and "values" in spec:
        val = trial.suggest_categorical(opt, spec["values"])
        sep = spec.get("sep", "=")
        return f"{opt}{sep}{val}", f"{opt}{sep}{val}"

    raise ValueError(f"Unsupported compiler_param entry for '{opt}'")

def _select_param_subset(trial, keys: List[str], sel: Dict[str, Any], max_k: Optional[int] = None) -> List[str]:
    if not sel:                 # default: include ALL params (old behaviour)
        return keys

    always = set(sel.get("always", [])) & set(keys)

    if "k" in sel:
        k = int(sel["k"])
    else:
        k = trial.suggest_int("params_k", int(sel.get("min", 0)), int(sel.get("max", len(keys))))

    if max_k is not None:
        required = int(sel.get("k", sel.get("min", 0)))
        effective_cap = max(int(max_k), required, len(always))
        k = min(k, effective_cap)

    # scores for a permutation; higher score = more likely to be chosen
    scored = []
    for kname in keys:
        if kname in always:
            scored.append((1.0, kname))    # force to the top
        else:
            s = trial.suggest_float(f"score::{kname}", 0.0, 1.0)
            scored.append((float(s), kname))

    scored.sort(reverse=True)              # by score
    chosen = [name for _, name in scored[:max(k, len(always))]]

    # Ensure all 'always' keys are included
    for a in always:
        if a not in chosen:
            chosen[-1] = a                 # replace the last one (k>=len(always) guaranteed)
    chosen_set = set(chosen)
    return [name for name in keys if name in chosen_set]


def suggest_compiler_flags(trial,
                           compiler_flags_base: str,
                           variants: List[str],
                           params_schema: Dict[str, Union[List[Any], Dict[str, Any]]],
                           flag_pool: List[str],
                           compiler_params_select: Optional[Dict[str, Any]] = None,
                           n_startup_trials: int = 10,
                           constraints: Optional[Any] = None,
                           ) -> Tuple[str, str]:
    
    chosen: List[str] = []
    label_parts: List[str] = []

    # ── 0. base flag

    chosen.append(compiler_flags_base)

    # ── 1. Variant strings
    variant: Optional[str] = None
    if variants:
        variant = trial.suggest_categorical("flag_variant", variants)
        chosen.append(variant)
        label_parts.append(variant)

    param_cap, pool_cap = _complexity_limits(
        trial,
        n_startup_trials=n_startup_trials,
        n_params_total=len(params_schema or {}),
        n_pool_total=len(flag_pool or []),
    )

    # ── 2. Parametric flags
    param_keys = list(params_schema.keys())
    active_keys = _select_param_subset(trial, param_keys, compiler_params_select or {}, max_k=param_cap)

    param_flags: List[str] = []
    for opt in active_keys:
        key_for_log, flag = _render_one_param(trial, opt, params_schema[opt])
        chosen.append(flag)
        label_parts.append(key_for_log)
        param_flags.append(flag)

    # ── 3. flag pool

    pool_list = list(flag_pool or [])
    if constraints is not None and not constraints.is_empty:
        # Everything the base, variant and params already activate (for implies).
        active_flags = [tok for tok in shlex.split(compiler_flags_base or "") if tok]
        if variant:
            active_flags.append(variant)
        active_flags.extend(param_flags)
        pool_list = constraints.prune_pool(pool_list, active_flags)
    else:
        active_flags = []

    picked_pool = _select_pool_flags(
        trial, pool_list, pool_cap,
        constraints=constraints, active=active_flags,
    )
    for f in picked_pool:
        chosen.append(f)
        label_parts.append(f)

    flag_str  = " ".join(chosen)
    pretty_id = "|".join(chosen) or "default"
    return pretty_id, flag_str



def suggest_env(trial, schema: Dict[str, Union[List[str], Dict[str, Any]]]
               ) -> Dict[str, str]:
    
    env: Dict[str, str] = {}

    # One linear pass is enough because predicates only look backwards
    # (already chosen vars).  If you need forward refs, convert to a loop.
    for var, spec in schema.items():
        # 1. Unconditional
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            env[var] = str(trial.suggest_categorical(var, list(spec)))
            continue

        # 2. Conditional
        pred  = spec.get("when", {})
        if all(env.get(k) == str(v) for k, v in pred.items()):
            env[var] = str(trial.suggest_categorical(var, spec["values"]))
        # else – silently skip

    return env

##### acpp cache cleaner 

def clear_acpp_runtime_cache(root: Optional[Path] = None) -> None:

    apps = root if root is not None else (Path.home() / ".acpp" / "apps")
    if not apps.exists():
        return

    # Only delete contents of 'apps', not the parent ~/.acpp directory.
    for entry in apps.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                # files or symlinks
                entry.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; ignore transient/permission errors
            pass

##### improvements


def rel_gain(old: float, new: float, goal: str) -> float:

    eps = 1e-12
    denom = max(abs(old), eps)
    if goal == "min":
        return (old - new) / denom
    else:
        return (new - old) / denom

def is_significant_improvement(old: float,
                               new: float,
                               goal: str,
                               min_rel_gain: float = 0.15,
                               min_abs_gain: Optional[float] = None) -> bool:

    absolute_gain = (old - new) if goal == "min" else (new - old)
    if absolute_gain <= 0:
        return False
    if min_abs_gain is not None and absolute_gain < float(min_abs_gain):
        return False
    return rel_gain(old, new, goal) >= float(min_rel_gain)

##### General helpers

def _normalize_args(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, Sequence):
        merged: List[str] = []
        for elem in raw:
            merged.extend(shlex.split(elem) if isinstance(elem, str) else [str(elem)])
        return merged
    raise TypeError("program_args must be string or list of strings")


def _expand_env(env: Dict[str, Sequence[str]]) -> List[EnvMap]:
    """
    Turn {"OMP_PLACES":["cores","sockets"], "ACPP_VISIBILITY_MASK":["a","b"]}
    into [{'OMP_PLACES':'cores','ACPP_VISIBILITY_MASK':'a'}, …]  (Cartesian product).
    """
    if not env:
        return [{}]
    # normalise: each value → list
    normd = {k: (v if isinstance(v, Sequence) and not isinstance(v, str) else [v])
             for k, v in env.items()}
    keys = list(normd)
    combos = itertools.product(*(normd[k] for k in keys))
    return [{k: str(val) for k, val in zip(keys, combo)} for combo in combos]

def unique_csv_path(path: Union[str, Path]) -> Path:
    """
    Return a writable CSV filename.
    If <path> does **not** exist → return as‑is.
    Otherwise append an ISO timestamp:  results.csv  → results_20250715‑1423.csv
    """
    p = Path(path)
    if not p.exists():
        return p

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    candidate = p.with_stem(f"{p.stem}_{stamp}")
    counter = 1
    while candidate.exists():
        candidate = p.with_stem(f"{p.stem}_{stamp}_{counter}")
        counter += 1
    return candidate

def output_prefixes(program_args: Sequence[str], extra_globs: Sequence[str] = ()) -> List[str]:
    """
    Infer the output-file prefixes a measured program writes into its run cwd.

    GROMACS names its outputs after ``-deffnm`` and refuses to overwrite them,
    which breaks re-measuring a configuration after an interrupted run.
    
    ``extra_globs`` lets a config add its own patterns.
    """
    globs: List[str] = [str(g) for g in extra_globs if str(g).strip()]
    args = _normalize_args(program_args)
    for index, token in enumerate(args):
        if token in {"-deffnm", "--deffnm"} and index + 1 < len(args):
            globs.append(f"{args[index + 1]}.*")
        elif token.startswith("-deffnm=") or token.startswith("--deffnm="):
            globs.append(f"{token.split('=', 1)[1]}.*")
    return list(dict.fromkeys(globs))

def clean_run_outputs(
    run_dir: Optional[Path],
    program_args: Sequence[str] = (),
    extra_globs: Sequence[str] = (),
) -> List[Path]:
    """
    Remove stale outputs of an interrupted measurement so it can be repeated.

    Only files matching inferred/configured patterns inside ``run_dir`` are
    touched; everything else (logs, build artifacts) is left alone.
    """
    if run_dir is None:
        return []
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    removed: List[Path] = []
    for pattern in output_prefixes(program_args, extra_globs):
        for path in sorted(run_dir.glob(pattern)):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                continue
    return removed

def save_run_output(stdout: str, stderr: str, *, prefix: str = "scout_run") -> Path:
    """
    Write a single run's stdout and stderr to a unique temp file and return its path.

    The file is created in the system tempdir via tempfile.mkstemp so it is
    unique per call even when many runs share the same working directory. The
    returned Path is meant to be surfaced in the terminal so a failed run can
    be inspected.
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".log")
    with os.fdopen(fd, "w") as fp:
        fp.write(stdout or "")
        if stderr:
            fp.write("\n--- stderr ---\n")
            fp.write(stderr)
    return Path(path)
