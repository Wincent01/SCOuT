from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import csv
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union


###############################################################################
# Third‑party imports                                                         #
###############################################################################

try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIIISampler, RandomSampler
    from optuna.trial import TrialState
except ImportError as exc:  # pragma: no cover
    sys.exit("[fatal] Optuna missing – install via `pip install optuna`.")

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import Config
from src.evaluator import Evaluator
from src.misc import suggest_compiler_flags, suggest_env, unique_csv_path

###############################################################################
# Optuna‑driven exploration                                                   #
###############################################################################

def explore_optuna(cfg: Config, n_trials: int) -> None:
    workdir_root = Path(tempfile.mkdtemp(prefix="SCOuT_"))
    print(f"[info] working directory root: {workdir_root}\n")
    evaluator = Evaluator(cfg, workdir_root)

    # Sampler choice
    is_multi = len(cfg.objectives) > 1
    complexity_startup = cfg.search.n_startup_trials
    if cfg.search.sampler == "nsga3":
        sampler = NSGAIIISampler(population_size=cfg.search.population_size,seed=cfg.search.random_seed,)
    elif cfg.search.sampler == "rs":
        sampler = RandomSampler(seed=cfg.search.random_seed,)
    else:
        startup = cfg.search.n_startup_trials or 0
        if is_multi and startup < 5:
            print("[info] MOTPE bootstrap: bumping n_startup_trials → 5")
            startup = 5
        complexity_startup = startup
        sampler = TPESampler(n_startup_trials=startup,
                            multivariate=True,
                            group=True,
                            seed=cfg.search.random_seed
                            )

    directions = ["minimize" if o.goal == "min" else "maximize" for o in cfg.objectives]
    storage = None
    if cfg.sqlite_log:
        sqlite_path = Path(cfg.sqlite_log)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{sqlite_path.resolve()}"
    study = optuna.create_study(sampler=sampler, directions=directions, storage=storage)

    # --- Optional: pick a Pareto CSV path (only for multi-objective) ---------
    pareto_path: Optional[Path] = None
    if is_multi:
        dest = cfg.pareto_log
        if dest:
            pareto_path = Path(dest)
        elif cfg.csv_log:
            base = Path(cfg.csv_log)
            pareto_path = base.with_name(f"{base.stem}_pareto.csv")
        else:
            pareto_path = workdir_root / "pareto.csv"

    def _export_pareto_front(study: optuna.study.Study, out_csv: Path) -> None:
        """Write current Pareto set to CSV (overwrites on each call)."""
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            front = study.best_trials  # Optuna ≥ 3.x
        except AttributeError:
            from optuna.visualization._pareto_front import _get_pareto_front_trials
            front = _get_pareto_front_trials(study)
        if not front:
            return
        # Collect any extra metric keys stored in user_attrs["metrics"].
        extra_metrics: set[str] = set()
        for t in front:
            extra_metrics.update((t.user_attrs.get("metrics") or {}).keys())
        extra_metrics.difference_update(o.metric for o in cfg.objectives)
        header = [o.metric for o in cfg.objectives] + ["compiler_flags", "env", "binary"] + sorted(extra_metrics)
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(header)
            for t in front:
                if t.values is None:
                    continue
                metrics: MetricDict = t.user_attrs.get("metrics", {})
                row = list(t.values) + [
                    t.user_attrs.get("compiler_flags", ""),
                    json.dumps(t.user_attrs.get("env", {})),
                    t.user_attrs.get("binary", ""),
                ]
                row += [metrics.get(k, "") for k in sorted(extra_metrics)]
                w.writerow(row)

    # Live-update Pareto CSV after each completed trial (multi-objective only)
    def _pareto_cb(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if is_multi and pareto_path is not None:
            _export_pareto_front(study, pareto_path)

    def trial_objective(trial: optuna.Trial):
        # --------------------------------------------------------------
        # 1) Sample discrete indices
        # --------------------------------------------------------------
        flag_key, flags = suggest_compiler_flags(
                            trial,
                            cfg.compiler_flags_base,
                            cfg.compiler_flags,
                            cfg.compiler_params,
                            cfg.compiler_flag_pool,
                            cfg.compiler_params_select,
                            complexity_startup,
                            constraints=cfg.flag_constraints,
                        )
        trial.set_user_attr("compiler_flags_str", flags)  

        env = suggest_env(trial, cfg.env)
        trial.set_user_attr("compiler_flags", flag_key)
        trial.set_user_attr("env", env)

        workdir = workdir_root / f"trial_{trial.number:05d}"
        workdir.mkdir()
        try:
            evaluation = evaluator.evaluate(flags, env, workdir)
            metrics = evaluation.metrics
            binary_path = evaluation.binary
        except Exception as exc:
            trial.set_user_attr("failure_reason", str(exc))
            raise optuna.TrialPruned(str(exc))
        

        # --------------------------------------------------------------
        # 4) Extract objective values (missing ⇒ prune)
        # --------------------------------------------------------------
        obj_values: List[Number] = []
        for obj in cfg.objectives:
            if obj.metric not in metrics:
                reason = f"metric '{obj.metric}' missing"
                trial.set_user_attr("failure_reason", reason)
                raise optuna.TrialPruned(reason)
            obj_values.append(metrics[obj.metric])

        # --------------------------------------------------------------
        # 5) Attach extra info for analysis
        # --------------------------------------------------------------

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("binary", str(binary_path))
        trial.set_user_attr("cached_evaluation", evaluation.cached)

        return obj_values

    study.optimize(trial_objective, n_trials=n_trials, show_progress_bar=True,
                   callbacks=([_pareto_cb] if is_multi and pareto_path is not None else None))


    # ------------------------------------------------------------------
    # Pareto front summary
    # ------------------------------------------------------------------
    try:
        front = study.best_trials          # Optuna ≥ 3.0
    except AttributeError:
        # Fallback for Optuna 2.x
        from optuna.visualization._pareto_front import _get_pareto_front_trials
        front = _get_pareto_front_trials(study)
    print("\n================ Pareto‑optimal configurations ================")
    for t in front:
        print(f"Trial#{t.number}: objectives={t.values} flags='{t.user_attrs['compiler_flags']}' env={t.user_attrs['env']}")
    print("==============================================================\n")
    
    if is_multi and pareto_path is not None:
        _export_pareto_front(study, pareto_path)
        print(f"[info] Pareto CSV written → {pareto_path}")

    # ------------------------------------------------------------------
    # CSV / SQLite logging
    # ------------------------------------------------------------------
    if cfg.csv_log:
        csv_path = unique_csv_path(cfg.csv_log)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[info] writing CSV log → {csv_path}")
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            # Header
            header = [o.metric for o in cfg.objectives] + ["compiler_flags", "env", "binary"]
            extra_metrics: set[str] = set()
            for t in study.trials:
                extra_metrics.update(t.user_attrs.get("metrics", {}).keys())
            extra_metrics.difference_update(o.metric for o in cfg.objectives)
            header += sorted(extra_metrics)
            writer.writerow(header)
            # Rows
            for t in study.trials:
                metrics: MetricDict = t.user_attrs.get("metrics", {})

                if t.state != TrialState.COMPLETE or t.values is None:
                    continue

                row = list(t.values) + [t.user_attrs.get("compiler_flags", ""), json.dumps(t.user_attrs.get("env", {})), t.user_attrs.get("binary","")]
                row += [metrics.get(k, "") for k in sorted(extra_metrics)]
                writer.writerow(row)

    failed = study.get_trials(states=(TrialState.FAIL, TrialState.PRUNED))
    if failed and cfg.fail_log:
        fail_path = Path(cfg.fail_log)
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["trial", "flags", "env", "reason", "binary"])
            for t in failed:
                writer.writerow([
                    t.number,
                    t.user_attrs.get("compiler_flags"),
                    json.dumps(t.user_attrs.get("env")),
                    t.user_attrs.get("failure_reason"),
                    t.user_attrs.get("binary"),
                ])
    

    if cfg.sqlite_log:
        print(f"[info] SQLite study written → {cfg.sqlite_log}")



###############################################################################
# Wavefront exploration                                                       #
###############################################################################

def explore_wavefront(cfg: Config) -> None:
    from src.searchMethods.wavefront_flags import run_wavefront_study
    return run_wavefront_study(cfg)

###############################################################################
# Tabu search exploration                                                     #
###############################################################################

def explore_tabu(
    cfg: Config,
    *,
    resume: bool = False,
    iters: Optional[int] = None,
    budget: str = "lifetime",
    workroot: Optional[Union[str, Path]] = None,
) -> None:
    from src.searchMethods.tabu_flags import run_tabu_study
    return run_tabu_study(cfg, resume=resume, iters=iters, budget=budget, workroot=workroot)

###############################################################################
# Beam and Tabu search exploration                                            #
###############################################################################

def explore_beam_tabu(cfg: Config) -> None:
    from src.searchMethods.beam_tabu import run_beam_tabu_study
    return run_beam_tabu_study(cfg)

###############################################################################
# Simulated Annealing                                                         #
###############################################################################

def explore_anneal(cfg: Config) -> None:
    from src.searchMethods.anneal_flags import run_anneal_study
    return run_anneal_study(cfg)
