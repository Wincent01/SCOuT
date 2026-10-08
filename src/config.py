from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import dataclasses
import json
import sys
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from src.misc import _normalize_args
from src.flag_constraints import FlagConstraints
from src.logger import LOG

###############################################################################
# Data‑classes for strongly‑typed config handling                             #
###############################################################################

@dataclasses.dataclass
class Objective:
    metric: str = "CPI"  # metric key to optimise
    goal: str = "min"    # "min" | "max"

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "Objective":
        goal = d.get("goal", "min").lower()
        if goal not in {"min", "max"}:
            raise ValueError("objective.goal must be 'min' or 'max'")
        return cls(metric=d.get("metric", "CPI"), goal=goal)

    # Note: better() is retained for callers that compare a single objective.

@dataclasses.dataclass
class PerfConfig:
    events: List[str]
    core_list: Optional[str]
    warmup_runs: int = 0
    

    @classmethod
    def from_dict(cls, d: Dict) -> "PerfConfig":
        return cls(
            events=d.get("events", ["cycles", "instructions"]),
            core_list=d.get("core_list"),
            warmup_runs=int(d.get("warmup_runs", 0))
        )


@dataclasses.dataclass
class MetricSpec:
    name: str                      # row name in LIKWID report
    agg: str = "avg"               # "avg" | "max" | "min" | "median"
    var: bool = False              # also compute variance?

    @classmethod
    def from_any(cls, raw):
        # Accept plain string for backward compat
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, dict):
            return cls(name=raw["name"],
                       agg=raw.get("agg", "avg").lower(),
                       var=bool(raw.get("var", False)))
        raise TypeError("metrics entries must be string or object")


@dataclasses.dataclass
class ParserConfig:
    # parser format
    format: str = "sycl"               # "sycl" | "gromacs"
    # which lines to use from the SYCL output
    label: str = "avg"                 # "avg" | "sum"
    # which kernel ids to consider (None = all ids found)
    kernels: Optional[List[int]] = None
    # how to aggregate across multiple kernels
    aggregate: str = "sum"             # "sum" | "mean" | "max" | "min"
    warmup_runs: int = 0

    # launch options to mimic perf/likwid
    core_list: Optional[str] = None    # e.g. "0-15" → taskset -c
    prefix: Optional[List[str]] = None # e.g. ["numactl","-N","0","-m","0"]
    run_cwd: str = "binary_dir"        # "binary_dir" | "project_dir" | "workdir"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParserConfig":
        return cls(
            format=d.get("format", "sycl"),
            label=d.get("label", "avg"),
            kernels=d.get("kernels"),
            aggregate=d.get("aggregate", "sum"),
            core_list=d.get("core_list"),
            prefix=d.get("prefix"),
            run_cwd=d.get("run_cwd", "binary_dir"),
            warmup_runs=int(d.get("warmup_runs", 0)),
        )

@dataclasses.dataclass
class LikwidConfig:
    group: str | None
    events: List[str] 
    metrics: List[MetricSpec]
    core_list: Optional[str]
    warmup_runs: int = 0

    @classmethod
    def from_dict(cls, d: Dict) -> "LikwidConfig":

        group   = d.get("group")
        events  = d.get("events", [])
        if isinstance(events, str):
            events = [e.strip() for e in events.split(",") if e.strip()]
        if not group and not events:
            raise ValueError("Need either 'group' or 'events' in likwid block")

        metrics_raw = d.get("metrics")                       # explicit list
        if not metrics_raw:                                  # or infer from events
            metrics_raw = [e.split(":")[0] for e in events]  if events else [group]
        metrics = [MetricSpec.from_any(m) for m in metrics_raw]

        return cls(
            group=group,
            events=events,
            metrics=metrics,
            core_list=d.get("core_list"),
            warmup_runs=int(d.get("warmup_runs", 0)),
        )


@dataclasses.dataclass
class BuildProject:
    dir: Path
    build_system: str = "cmake"  # "cmake" | "make"
    target: Optional[str] = None
    build_jobs: Optional[int] = None
    executable: Optional[Path] = None
    make_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    make_flags_var: str = "CXXFLAGS"

    cmake_defs: List[str] = dataclasses.field(default_factory=list)
    cmake_flag_vars: List[str] = dataclasses.field(       
        default_factory=lambda: ["CMAKE_CXX_FLAGS"]
    )

    @classmethod
    def from_dict(cls, d: Dict) -> "BuildProject":
        build_system = str(d.get("build_system", "cmake")).lower()
        if build_system not in {"cmake", "make"}:
            raise ValueError("project.build_system must be 'cmake' or 'make'")
        return cls(
            dir=Path(d["dir"]),
            build_system=build_system,
            target=d.get("target"),
            build_jobs=int(d["build_jobs"]) if d.get("build_jobs") is not None else None,
            executable=Path(d["executable"]) if d.get("executable") else None,
            make_vars=d.get("make_vars", {}),
            make_flags_var=d.get("make_flags_var", "CXXFLAGS"),
            cmake_flag_vars=d.get("cmake_flag_vars", ["CMAKE_CXX_FLAGS"]),
            cmake_defs=d.get("cmake_defs", []),
        )


@dataclasses.dataclass
class SearchSpec:
    study: str = "optuna"  # only optuna for now
    sampler: str = "tpe"     # "tpe" | "nsga3" | "rs"
    n_startup_trials: int = 10  # for TPE
    population_size: int = 50
    random_seed: int | None = None

    @classmethod
    def from_dict(cls, d: Dict) -> "SearchSpec":
        study = str(d.get("study", "optuna")).lower()
        sampler = str(d.get("sampler", "tpe")).lower()
        valid_studies = {"optuna", "wavefront", "tabu", "beam_tabu", "anneal"}
        valid_samplers = {"tpe", "nsga3", "rs"}
        if study not in valid_studies:
            raise ValueError(f"search.study must be one of {sorted(valid_studies)}")
        if sampler not in valid_samplers:
            raise ValueError(f"search.sampler must be one of {sorted(valid_samplers)}")
        return cls(
            study=study,
            sampler=sampler,
            n_startup_trials=int(d.get("n_startup_trials", 10)),
            population_size=int(d.get("population_size", 50)),
            random_seed=d.get("random_seed"),
        )

@dataclasses.dataclass
class WavefrontSpec:
    base_flags: List[str] = dataclasses.field(default_factory=list)
    flag_atoms: Optional[List[str]] = None
    max_k: int = 3
    mode: str = "beam"         # "beam" | "full"
    beam_width: int = 16
    per_wave_cap: Optional[int] = None
    stop_if_no_improve: bool = True
    improvement_eps: float = 0.0
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    env_mode: str = "product"
    env_cap: Optional[int] = None
    results_csv: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "WavefrontSpec":
        env_mode = str(d.get("env_mode", "product")).lower()
        if env_mode not in {"fixed", "product", "sample"}:
            raise ValueError("wavefront.env_mode must be 'fixed', 'product', or 'sample'")
        return cls(
            base_flags=d.get("base_flags", []),
            flag_atoms=d.get("flag_atoms"),
            max_k=int(d.get("max_k", 3)),
            mode=d.get("mode", "beam"),
            beam_width=int(d.get("beam_width", 16)),
            per_wave_cap=d.get("per_wave_cap"),
            stop_if_no_improve=bool(d.get("stop_if_no_improve", True)),
            improvement_eps=float(d.get("improvement_eps", 0.0)),
            env=d.get("env", {}) or {},
            env_mode=env_mode,
            env_cap=(None if d.get("env_cap") is None else int(d["env_cap"])),
            results_csv=d.get("results_csv")
        )


@dataclasses.dataclass
class SignificanceSpec:
    min_rel_gain: float = 0.0
    min_abs_gain: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignificanceSpec":
        return cls(
            min_rel_gain=float(d.get("min_rel_gain", 0.0)),
            min_abs_gain=(None if d.get("min_abs_gain") is None else float(d["min_abs_gain"])),
        )


@dataclasses.dataclass
class PolyMorphSearchSpec:
    DEFAULT_BLOCK_TRANSFORMS = ["SET_PARALLEL"]

    n_trials: int = 20
    repeat: int = 3
    generated_infix: str = "mcts"
    seed: int = 0
    timeout: int | None = None
    enumerate_only: bool = False
    baseline_exec_name: str | None = None
    max_transforms_per_trial: int = 1
    tile_sizes: List[int] = dataclasses.field(default_factory=lambda: [8, 16, 32, 64])
    allow_transforms: List[str] | None = None
    block_transforms: List[str] = dataclasses.field(
        default_factory=lambda: PolyMorphSearchSpec.DEFAULT_BLOCK_TRANSFORMS[:]
    )
    result_json: str | None = None
    trial_csv: str | None = None
    static_pruning: bool = False
    analytical_model: bool = False
    constraint_aware: bool = False
    case_retrieval: bool = False
    structural_retrieval: bool = True
    cache_jsonl: str | None = None
    cache_evaluations: bool = True
    multi_fidelity: bool = False
    early_stop_worse_than: float = 1.15
    top_k: int | None = None
    retrieval_top_k: int = 3
    history_jsonl: str | None = None
    pareto_csv: str | None = None
    constraints: Dict[str, Any] = dataclasses.field(default_factory=dict)
    backend_sensitivity_masks: List[str] = dataclasses.field(default_factory=list)
    backend_sensitivity_repeat: int = 1
    backend_sensitivity_per_trial: bool = False
    legality_aware_args: bool = True
    correctness_outputs: List[str] = dataclasses.field(default_factory=list)
    correctness_tolerance: float = 1.0e-6
    correctness_required: bool = True
    learned_model: bool = True
    learned_model_min_observations: int = 1
    target_backend: str | None = None
    replay_top_k: int = 0
    final_validation_enabled: bool = True
    final_validation_repeats: int = 20
    final_validation_warmup_runs: int = 1
    trial_warmup_runs: int = 1
    final_validation_top_k: int = 3
    final_validation_min_speedup: float = 1.0
    final_validation_noise_factor: float = 2.0
    baseline_resample_interval: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "PolyMorphSearchSpec":
        data = raw or {}
        allow = data.get("allow_transforms")
        block = data.get("block_transforms", cls.DEFAULT_BLOCK_TRANSFORMS)
        return cls(
            n_trials=int(data.get("n_trials", 20)),
            repeat=int(data.get("repeat", 3)),
            generated_infix=str(data.get("generated_infix", "mcts")),
            seed=int(data.get("seed", 0)),
            timeout=int(data["timeout"]) if data.get("timeout") is not None else None,
            enumerate_only=bool(data.get("enumerate_only", False)),
            baseline_exec_name=data.get("baseline_exec_name"),
            max_transforms_per_trial=int(data.get("max_transforms_per_trial", 1)),
            tile_sizes=[int(x) for x in data.get("tile_sizes", [8, 16, 32, 64])],
            allow_transforms=[str(x) for x in allow] if allow is not None else None,
            block_transforms=[str(x) for x in block],
            result_json=data.get("result_json"),
            trial_csv=data.get("trial_csv"),
            static_pruning=bool(data.get("static_pruning", False)),
            analytical_model=bool(data.get("analytical_model", False)),
            constraint_aware=bool(data.get("constraint_aware", False)),
            case_retrieval=bool(data.get("case_retrieval", False)),
            structural_retrieval=bool(data.get("structural_retrieval", True)),
            cache_jsonl=data.get("cache_jsonl"),
            cache_evaluations=bool(data.get("cache_evaluations", True)),
            multi_fidelity=bool(data.get("multi_fidelity", False)),
            early_stop_worse_than=float(data.get("early_stop_worse_than", 1.15)),
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            retrieval_top_k=int(data.get("retrieval_top_k", 3)),
            history_jsonl=data.get("history_jsonl"),
            pareto_csv=data.get("pareto_csv"),
            constraints=dict(data.get("constraints", {}) or {}),
            backend_sensitivity_masks=[
                str(mask) for mask in data.get("backend_sensitivity_masks", [])
                if str(mask).strip()
            ],
            backend_sensitivity_repeat=int(data.get("backend_sensitivity_repeat", 1)),
            backend_sensitivity_per_trial=bool(data.get("backend_sensitivity_per_trial", False)),
            legality_aware_args=bool(data.get("legality_aware_args", True)),
            correctness_outputs=[str(path) for path in data.get("correctness_outputs", [])],
            correctness_tolerance=float(data.get("correctness_tolerance", 1.0e-6)),
            correctness_required=bool(data.get("correctness_required", True)),
            learned_model=bool(data.get("learned_model", True)),
            learned_model_min_observations=int(data.get("learned_model_min_observations", 1)),
            target_backend=str(data["target_backend"]) if data.get("target_backend") is not None else None,
            replay_top_k=int(data.get("replay_top_k", 0)),
            final_validation_enabled=bool(data.get("final_validation_enabled", True)),
            final_validation_repeats=int(data.get("final_validation_repeats", 20)),
            final_validation_warmup_runs=int(data.get("final_validation_warmup_runs", 1)),
            trial_warmup_runs=int(data.get("trial_warmup_runs", data.get("final_validation_warmup_runs", 1))),
            final_validation_top_k=int(data.get("final_validation_top_k", 3)),
            final_validation_min_speedup=float(data.get("final_validation_min_speedup", 1.0)),
            final_validation_noise_factor=float(data.get("final_validation_noise_factor", 2.0)),
            baseline_resample_interval=int(data.get("baseline_resample_interval", 0)),
        )


@dataclasses.dataclass
class PolyMorphSpec:
    project_root: Path
    source: Optional[Path] = None
    compiler: str = "acpp"
    flags: List[str] = dataclasses.field(default_factory=list)
    transforms: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    allow_illegal: bool = False
    print_available_transformations: bool = False
    list_only: bool = False
    save_jscops: Optional[Path] = None
    discover: bool = False
    mcts_search: bool = False
    build_system: str = "make"
    build_target: Optional[str] = None
    build_dir: Optional[Path] = None
    exec_name: Optional[str] = None
    runtime_args: List[str] = dataclasses.field(default_factory=list)
    measure: bool = False
    generated_infix: str = "pass1"
    search: PolyMorphSearchSpec = dataclasses.field(default_factory=PolyMorphSearchSpec)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PolyMorphSpec":
        source_raw = raw.get("source")
        save_jscops_raw = raw.get("save_jscops")
        build_dir_raw = raw.get("build_dir")
        transforms = raw.get("transforms", []) or []
        flags = raw.get("flags", []) or []
        build_system = str(raw.get("build_system", "make"))
        if build_system not in {"make", "cmake"}:
            raise ValueError("polyMorph.build_system must be 'make' or 'cmake'")
        if not raw.get("project_root"):
            raise ValueError("polyMorph.project_root is required")
        if not isinstance(transforms, list):
            raise ValueError("polyMorph.transforms must be a list")
        if not isinstance(flags, list):
            raise ValueError("polyMorph.flags must be a list")

        return cls(
            project_root=Path(raw["project_root"]).expanduser().resolve(),
            source=Path(source_raw).expanduser().resolve() if source_raw else None,
            compiler=str(raw.get("compiler", "acpp")),
            flags=[str(flag) for flag in flags if str(flag).strip()],
            transforms=[dict(item) for item in transforms],
            allow_illegal=bool(raw.get("allow_illegal", False)),
            print_available_transformations=bool(raw.get("print_available_transformations", False)),
            list_only=bool(raw.get("list_only", False)),
            save_jscops=Path(save_jscops_raw).expanduser().resolve() if save_jscops_raw else None,
            discover=bool(raw.get("discover", False)),
            mcts_search=bool(raw.get("mcts_search", False)),
            build_system=build_system,
            build_target=raw.get("build_target"),
            build_dir=Path(build_dir_raw).expanduser().resolve() if build_dir_raw else None,
            exec_name=raw.get("exec_name"),
            runtime_args=_normalize_args(raw.get("runtime_args")),
            measure=bool(raw.get("measure", False)),
            generated_infix=str(raw.get("generated_infix", "pass1")),
            search=PolyMorphSearchSpec.from_dict(raw.get("search")),
        )


@dataclasses.dataclass
class Config:
    backend: str  # "perf" | "likwid"

    # Build description (one of the two)
    source: Optional[Path]
    project: Optional[BuildProject]

    # Program arguments and environment sets
    program_args: List[str] 

    # Compiler flags
    compiler: str 
    compiler_flags_base: str 
    compiler_flags: List[str] 
    compiler_params: Dict[str, Union[List[Any], Dict[str, Any]]] 
    compiler_params_select: Dict[str, Any] 
    compiler_flag_pool: List[str]
    
    #Environment variables
    env: Dict[str, Union[List[str], Dict[str, Any]]] 

    # Backend‑specific blocks
    perf: Optional[PerfConfig]
    likwid: Optional[LikwidConfig]
    parser: Optional[ParserConfig] 

    # Objectives (≥1)
    objectives: List[Objective] 

    # Search algorithm details
    search: SearchSpec

    # Whether AdaptiveCpp's runtime/JIT cache is reused or deliberately cleared.
    runtime_cache_policy: str
    significance: SignificanceSpec

    runs: int
    # CSV / SQLite log paths
    csv_log: Optional[str]
    pareto_log: Optional[str]
    fail_log: Optional[str]
    sqlite_log: Optional[str]

    # Build/run/runtime statistics CSV (default: ./scout_stats.csv)
    stats_log: Optional[str] = None

    # Declarative rules over flag combinations (exclusive groups / implied flags)
    flag_constraints: FlagConstraints = dataclasses.field(default_factory=FlagConstraints)

    # Wavefront
    wavefront: Optional[WavefrontSpec] = None
    #tabu
    tabu: Dict[str, Any] = dataclasses.field(default_factory=dict)
    beam_tabu: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #anneal 
    anneal: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # polyMorph
    poly_morph: Optional[PolyMorphSpec] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> "Config":
        with open(path, "r", encoding="utf-8") as fp:
            raw = json.load(fp)

        poly_morph_raw = raw.get("polyMorph")
        poly_morph = PolyMorphSpec.from_dict(poly_morph_raw) if isinstance(poly_morph_raw, dict) else None

        backend = raw.get("backend", "likwid").lower()
        if backend not in {"perf", "likwid", "parser"}:
            raise ValueError("backend must be 'perf' or 'likwid' or 'parser'")

        # Build description
        source = raw.get("source")
        project = raw.get("project")
        if bool(source) == bool(project) and poly_morph is None:
            raise ValueError("Provide exactly one of 'source' or 'project', or define 'polyMorph'.")
        
        # Compiler flags
        def _validate_params_select(sel: Dict[str, Any], available: List[str]) -> Dict[str, Any]:
            if not sel:
                return {}
            if "k" in sel and ("min" in sel or "max" in sel):
                raise ValueError("compiler_params_select: use either 'k' or 'min/max', not both")
            if "always" in sel:
                unknown = [k for k in sel["always"] if k not in available]
                if unknown:
                    raise ValueError(f"compiler_params_select.always has unknown keys: {unknown}")
            for bound in ("k", "min", "max"):
                if bound in sel and not 0 <= int(sel[bound]) <= len(available):
                    raise ValueError(f"compiler_params_select.{bound} must be between 0 and {len(available)}")
            if int(sel.get("min", 0)) > int(sel.get("max", len(available))):
                raise ValueError("compiler_params_select.min cannot exceed max")
            required = int(sel.get("k", sel.get("max", len(available))))
            if len(sel.get("always", [])) > required:
                raise ValueError("compiler_params_select.always contains more entries than the selected count permits")
            return sel
        
        compiler_params = raw.get("compiler_params", {})
        if not isinstance(compiler_params, dict):
            raise ValueError("compiler_params must be an object")
        for name, spec in compiler_params.items():
            values = spec.get("values") if isinstance(spec, dict) else spec
            if not isinstance(values, list) or not values:
                raise ValueError(f"compiler_params.{name} must have a non-empty value list")
        compiler_params_select = _validate_params_select(
            raw.get("compiler_params_select", {}),
            list(compiler_params.keys())
        )

        # Declarative flag constraints (exclusive groups / implied flags)
        flag_constraints = FlagConstraints.from_dict(raw.get("flag_constraints"))

        # Environment space
        env_schema = raw.get("env")
        if not isinstance(env_schema, dict):
            env_schema = {}
        
        for var, spec in env_schema.items():
            if isinstance(spec, list):
                if not spec:
                    raise ValueError(f"'env.{var}' must contain at least one value")
                continue
            if not (isinstance(spec, dict) and "values" in spec):
                raise ValueError(
                    f"'env.{var}' must be a list or an object with a 'values' key"
                )
            if not isinstance(spec["values"], list) or not spec["values"]:
                raise ValueError(f"'env.{var}.values' must be a non-empty list")
            unknown_predecessors = [k for k in spec.get("when", {}) if k not in list(env_schema)[:list(env_schema).index(var)]]
            if unknown_predecessors:
                raise ValueError(
                    f"env.{var}.when may only reference earlier environment variables; "
                    f"invalid references: {unknown_predecessors}"
                )

        # Program arguments
        program_args = _normalize_args(raw.get("program_args"))

        # Objectives
        objs_raw = raw.get("objectives")
        if not objs_raw:
            # Fallback to single objective block from backend
            if backend == "perf":
                o_raw = raw.get("perf", {}).get("objective", {})
            elif backend == "likwid":
                o_raw = raw.get("likwid", {}).get("objective", {})
            else:
                parser_raw = raw.get("parser", {})
                label = str(parser_raw.get("label", "avg"))
                aggregate = str(parser_raw.get("aggregate", "sum"))
                o_raw = parser_raw.get("objective", {"metric": f"sycl_{label}_{aggregate}_s", "goal": "min"})
            objs_raw = [o_raw]

        objectives = [Objective.from_dict(o) for o in objs_raw]
        if not objectives:
            raise ValueError("At least one objective required.")
        
        # Parser
        parser_cfg = ParserConfig.from_dict(raw.get("parser", {})) if backend == "parser" else None

        tabu = raw.get("tabu", {})

        anneal = raw.get("anneal", {})

        for block_name, block in (("tabu", tabu), ("anneal", anneal), ("beam_tabu", raw.get("beam_tabu", {}))):
            if not isinstance(block, dict):
                raise ValueError(f"{block_name} must be an object")
            env_mode = str(block.get("env_mode", "product")).lower()
            if env_mode not in {"fixed", "product", "sample"}:
                raise ValueError(f"{block_name}.env_mode must be 'fixed', 'product', or 'sample'")

        search = SearchSpec.from_dict(raw.get("search", {}))
        if search.study != "optuna" and len(objectives) != 1:
            raise ValueError(f"search.study='{search.study}' supports exactly one objective")
        runtime_cache_policy = str(raw.get("runtime_cache_policy", "reuse")).lower()
        if runtime_cache_policy not in {"reuse", "cold"}:
            raise ValueError("runtime_cache_policy must be 'reuse' or 'cold'")

        runs = int(raw.get("runs", 1))
        if runs <= 0:
            raise ValueError("runs must be a positive integer")

        loaded_project = BuildProject.from_dict(project) if project else None
        loaded_source = Path(source) if source else None
        if loaded_source is not None and not loaded_source.is_file():
            raise ValueError(f"source does not exist: {loaded_source}")
        if loaded_project is not None and not loaded_project.dir.is_dir():
            raise ValueError(f"project.dir does not exist: {loaded_project.dir}")
        if loaded_project is not None and loaded_project.executable is None:
            raise ValueError(
                "project.executable is required; build target names such as 'all' "
                "do not identify the runnable file"
            )

        if not flag_constraints.is_empty:
            raw_flags = raw.get("compiler_flags", []) or []
            if isinstance(raw_flags, dict):
                raw_flags = list(raw_flags.values())
            universe = {str(flag) for flag in raw_flags}
            universe |= {str(flag) for flag in (raw.get("compiler_flag_pool", []) or [])}
            for message in flag_constraints.validate_against(universe):
                LOG.log(f"[config] warning: {message}")

        return cls(
            backend=backend,
            source=loaded_source,
            project=loaded_project,
            compiler=raw.get("compiler", "acpp"),
            compiler_flags_base=raw.get("compiler_flags_base", ""),
            compiler_flags=raw.get("compiler_flags", []),
            compiler_params = compiler_params,
            compiler_params_select = compiler_params_select,
            compiler_flag_pool  = raw.get("compiler_flag_pool", []),
            program_args=program_args,
            env=env_schema,
            perf=PerfConfig.from_dict(raw.get("perf", {})) if backend == "perf" else None,
            likwid=LikwidConfig.from_dict(raw.get("likwid", {})) if backend == "likwid" else None,
            parser=parser_cfg,
            objectives=objectives,
            search=search,
            runtime_cache_policy=runtime_cache_policy,
            significance=SignificanceSpec.from_dict(raw.get("significance", {})),
            wavefront=WavefrontSpec.from_dict(raw.get("wavefront", {})) if "wavefront" in raw else None,
            tabu=tabu,
            beam_tabu=raw.get("beam_tabu", {}) or {},
            anneal=anneal,
            poly_morph=poly_morph,
            flag_constraints=flag_constraints,
            runs=runs,
            csv_log=raw.get("csv_log"),
            pareto_log=raw.get("pareto_log"),
            fail_log=raw.get("failed_builds"),
            sqlite_log=raw.get("sqlite_log"),
            stats_log=raw.get("stats_log"),
        )
