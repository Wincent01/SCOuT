# Parameter tuning guide

This guide covers SCOuT's compiler-flag and runtime-environment tuning mode. It does not cover PolyMorph.

## Install

Use Python 3.10 or newer and install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install the measurement tool used by your configuration (`perf` or LIKWID). The `parser` backend needs no external counter tool, but the benchmark must print the documented `[SYCL]` timing lines.

## Run

Run repository-provided configurations from the repository root so their relative benchmark paths resolve correctly.

```bash
python3 main.py \
  --mode parameter_tuning \
  --config configs/flag-env/optuna_random_perf.json \
  --trials 50
```

`--trials` controls Optuna. Wavefront, tabu, beam-tabu, and annealing use the limits in their own configuration blocks.

## Project builds

Keep the build target and runnable file separate:

```json
"project": {
  "dir": "test-benchmarks/parameterSearch/microSYCL",
  "build_system": "make",
  "target": "all",
  "executable": "bin/main-acpp-generic",
  "make_flags_var": "EXTRA_CFLAGS",
  "make_vars": {
    "BACKEND": "generic"
  }
}
```

- `target` is passed to Make or CMake.
- `executable` is the file SCOuT runs after the build. For Make it is relative to `project.dir`; for CMake it is relative to the generated build directory.
- Always set `executable` when the target is phony, such as `all`.

For a CMake project, use `build_system: "cmake"`:

```json
"project": {
  "dir": "path/to/project",
  "build_system": "cmake",
  "target": "my_program",
  "executable": "bin/my_program",
  "cmake_flag_vars": ["CMAKE_CXX_FLAGS"],
  "cmake_defs": [
    "BUILD_TESTING=OFF",
    "MY_BACKEND=CPU"
  ]
}
```

For every distinct compiler configuration, SCOuT performs the equivalent of:

```bash
cmake -S path/to/project -B <workdir> \
  -DCMAKE_CXX_COMPILER=<compiler> \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS:STRING=<sampled-flags> \
  -DBUILD_TESTING=OFF \
  -DMY_BACKEND=CPU
cmake --build <workdir> --parallel --target my_program
```

- `compiler` becomes `CMAKE_CXX_COMPILER`.
- Each `cmake_defs` entry becomes a `-D` CMake definition.
- Every variable in `cmake_flag_vars` receives the complete generated flag string.
- `executable` is relative to SCOuT's generated CMake build directory, not `project.dir`. Use the path CMake generates, such as `my_program`, `bin/my_program`, or `Release/my_program.exe`.
- If a project exposes its own flag cache variable, use that instead of `CMAKE_CXX_FLAGS`. The miniBUDE example uses `CXX_EXTRA_FLAGS` for this reason.

A complete repository-backed example is available at
[`configs/example_configs/cmake_project.json`](../configs/example_configs/cmake_project.json).
Run it from the repository root:

```bash
python3 main.py --mode parameter_tuning \
  --config configs/example_configs/cmake_project.json \
  --trials 10
```

The example expects the AdaptiveCpp CMake package under `/opt/AdaptiveCpp`, as
required by the included miniBUDE `CMakeLists.txt`. If your installation is
elsewhere, update that benchmark's `HIPSYCL_INSTALL_DIR` setting or use your own
CMake project and provide its package-location definition in `cmake_defs`.

For a single source, use `"source": "path/to/file.cpp"` instead of `project`.

## Compiler and runtime spaces

```json
"compiler_flags_base": "--acpp-platform=cpu --acpp-targets=generic",
"compiler_flags": ["-O2", "-O3"],
"compiler_flag_pool": ["-fvectorize", "-ffast-math"],
"compiler_params": {
  "-march": ["native", "znver4"],
  "--tile={}": [8, 16, 32]
},
"compiler_params_select": {
  "min": 1,
  "max": 2,
  "always": ["-march"]
},
"env": {
  "ACPP_VISIBILITY_MASK": ["omp", "generic"],
  "OMP_PLACES": {
    "when": {"ACPP_VISIBILITY_MASK": "omp"},
    "values": ["cores", "threads"]
  }
}
```

Conditional environment variables must appear after every variable referenced by `when`. SCOuT removes managed-but-inactive variables from the inherited process environment, preventing shell settings from leaking into a candidate.

### Flag constraints

Searches that combine compiler flags can waste evaluations on combinations that cancel each other
out or repeat a flag an optimization level already enables. Declare those rules once in a top-level
`flag_constraints` block:

```json
"flag_constraints": {
  "exclusive": [
    ["-funroll-loops", "-fno-unroll-loops"],
    ["-fstrict-aliasing", "-fno-strict-aliasing"],
    ["-fexceptions", "-fno-exceptions"]
  ],
  "implies": {
    "-ffast-math": ["-fno-math-errno", "-ffp-contract=fast"],
    "-O2": ["-fstrict-aliasing", "-fdelete-null-pointer-checks", "-fexceptions"],
    "-O3": ["-fstrict-aliasing", "-fdelete-null-pointer-checks", "-fexceptions", "-funroll-loops"]
  }
}
```

- `exclusive` lists groups of which **at most one** flag may be active at a time. The search never
  emits a configuration containing two members of the same group. Members of an exclusive group are
  part of the flag space implicitly, so they do not have to be listed in `compiler_flag_pool` too
  (unlike `implies` values, which must be listed explicitly).
- `implies` maps an active flag (a `compiler_flags` entry such as `-O3`, or a pool flag such as
  `-ffast-math`) to the flags it already enables. Those flags are **redundant** while the source flag
  is active. The relation is applied transitively.

Flags are matched by exact string equality against entries of `compiler_flags` and
`compiler_flag_pool` (and rendered `compiler_params`). Referencing a flag that is not in the space
only logs a warning.

`implies` is removal-only: an implied flag is dropped, but it is *not* treated as active. This keeps
override experiments testable — with `-O3` implying `-funroll-loops`, `-O3 -funroll-loops` is
redundant and skipped while `-O3 -fno-unroll-loops` is still explored.

### Equivalent configurations in tabu

`tabu` keeps implied flags in the search space rather than deleting them: a configuration such as
`-O3 -funroll-loops` is still a node the search can move to. Before building anything, the requested
flags are reduced to their effective form (implied flags removed); when that effective configuration
was already measured, the earlier result is reused and nothing is built or measured again. Two flag
lists are only treated as equivalent when they have the same effective flags *and* the same
environment, so a genuinely different configuration is never skipped.

Because such a node adds no new information it does not count as an iteration: neither the iteration
budget nor the no-improvement counter advances. `tabu.max_skips` (default `50`) bounds how many
consecutive equivalent nodes may be traversed before the search stops with
`consecutive equivalent nodes`.

`optuna` and the other searches prune implied flags from their space instead. The rules are part of
the tabu resume fingerprint, so changing them invalidates an existing `--resume` checkpoint. The
flags implied by `-O2`/`-O3` are compiler-specific, confirm the mapping against your toolchain.

## Build and AdaptiveCpp cache reuse

SCOuT caches these separately:

- Build cache: compiler, compiler flags, source/project, and build settings.
- Evaluation cache: build key plus runtime environment.

Changing only an environment variable therefore reuses the existing executable. If the compiler space contains one configuration, SCOuT builds exactly once during that process.

AdaptiveCpp runtime/JIT cache behavior is explicit:

```json
"runtime_cache_policy": "reuse"
```

- `reuse` is the default and avoids repeated runtime compilation.
- `cold` clears `~/.acpp/apps` after each measured configuration. Use it only when intentionally studying cold-start behavior; it affects the user's complete AdaptiveCpp application cache.

For large runtime spaces, set an environment cap in a custom search, for example `"env_mode": "sample", "env_cap": 16`. A fixed environment is written directly in that search block using `"env": {...}`.

## Search methods

- `optuna`: supports single- and multi-objective studies. Use `tpe`, `rs`, or `nsga3` for SCOuT's categorical space.
- `wavefront`: explores flag-atom combinations by size.
- `tabu`: moves among variants, compiler parameters, pool flags, and environments.
- `beam_tabu`: retains multiple candidates and uses tabu move memory.
- `anneal`: stochastic local search with seeded acceptance decisions.

Custom searches currently require exactly one objective. SCOuT rejects incompatible configurations rather than creating malformed result files.

Example search selection:

```json
"search": {
  "study": "tabu",
  "random_seed": 13
},
"tabu": {
  "max_iters": 40,
  "max_no_improve": 10,
  "tabu_tenure": 12,
  "neighborhood": 16,
  "env_mode": "sample",
  "env_cap": 8,
  "results_csv": "/tmp/scout/tabu.csv",
  "state_file": "/tmp/scout/tabu.state.json",
  "clean_run_outputs": true,
  "clean_globs": ["mdout.mdp"]
}
```

`results_csv`, `state_file`, `failure_csv`, `run_dir`, `clean_run_outputs` and `clean_globs` control
the durable run record described under [Resuming a tabu run](#resuming-a-tabu-run); the remaining keys
shape the search itself.

Unknown study and sampler names are configuration errors. CMA-ES is not offered because SCOuT's flag and environment dimensions are categorical; use TPE, random sampling, or NSGA-III.

## Resuming a tabu run

Tabu is built for clusters where a single job cannot hold the whole search. It writes every
completed measurement to the results CSV immediately and snapshots its search state after each
completed iteration, so an interrupted run (walltime kill, node failure, `Ctrl-C`) can be continued
instead of restarted.

| File | Purpose |
| --- | --- |
| `<results_csv>` | Every measured configuration: `k`, objective metrics, `compiler_flags`, `env`, `binary`, `eval_id`, `flags`, `compile_key`, `cached`, plus one column per extra metric. Appended and flushed per measurement. |
| `<results_csv>.state.json` | Atomic snapshot of the search: current configuration, tabu memory, best result, iteration and no-improvement counters, RNG state, sampled environment list, phase. |
| `failed_builds` (or `tabu.failure_csv`) | Configurations that failed to build or measure, with the reason. Skipped on later attempts. |
| `<results_csv>.archive-<timestamp>/` | Previous CSV, state and failures, moved aside when a new run starts. |

CLI options:

```bash
python3 main.py --mode parameter_tuning --config configs/flag-env/tabu_gromacs_cpu_omp.json \
  --trials 1 --iters 40 --budget lifetime
```

- `--resume` continues the run recorded in the configured results CSV. Without it, a new run starts and any earlier files are archived rather than overwritten.
- `--fresh` states that intent explicitly (it is the default).
- `--iters N` overrides `tabu.max_iters`.
- `--budget lifetime` (default) treats `N` as the total number of iterations the run may perform across all restarts; `--budget per-run` grants `N` additional iterations on top of the completed ones.
- `--workroot DIR` chooses where builds and run scratch live, defaulting to `$SCOUT_TABU_WORKROOT`, then `tabu.run_dir`, then `$TMPDIR/SCOuT_tabu`.

Submit, get killed, resubmit:

```bash
#SBATCH -t 01:00:00
python3 main.py --mode parameter_tuning --config configs/flag-env/tabu_gromacs_cpu_omp.json \
  --trials 1 --iters 40 --resume
```

What is reused on resume:

- Configurations already in the results CSV are never built or measured again; their recorded metrics are restored as the search state.
- Recorded failures are skipped, so a configuration that cannot build does not consume job time twice.
- Build artifacts in the workroot are reused when they are still there. Nothing is assumed to survive: if the workroot was wiped (a different node, a cleaned `$TMPDIR`), those configurations are rebuilt from scratch while the recorded results still come from the CSV.
- `tabu.clean_run_outputs` (default `true`) removes leftover program outputs named after `-deffnm` before re-measuring a configuration, which is what lets an interrupted measurement be repeated when the program refuses to overwrite its own output. Add extra patterns with `tabu.clean_globs`.

Resume is refused when the configuration no longer matches the checkpoint (compiler, flags, params,
pool, environment schema, objectives, program arguments, build settings, or the tabu move options).
Iteration limits, tenure and neighborhood are deliberately excluded so budgets can be adjusted
between restarts.

## Measurement backends

For `perf`, commonly use:

```json
"perf": {
  "events": ["cycles", "instructions"],
  "warmup_runs": 1
},
"objectives": [{"metric": "CPI", "goal": "min"}]
```

### Working directory (`run_cwd`)

All three backends select the directory the measured program runs in with `run_cwd`:

```json
"perf": {
  "events": ["cycles", "instructions"],
  "run_cwd": "binary_dir"
}
```

- `process` (default for `perf`/`likwid`) inherits SCOuT's own working directory.
- `binary_dir` runs the program next to the built executable and is the `parser` default. Use it
  when the workload writes result files into the current directory, so they land in the per-build
  directory instead of wherever SCOuT was launched.
- `project_dir` uses the configured project directory.
- `workdir` uses the per-evaluation scratch directory (`runs/<key>/<env>`), which is also where
  tabu removes stale outputs before re-measuring.

Relative paths in `program_args` are resolved against the chosen directory. The `parser` backend
additionally accepts `prefix`.

For the parser backend, a program line looks like:

```text
[SYCL][avg] kernel 2: 1.25e-06 s over 1000 iters
```

The aggregate objective is named `sycl_<label>_<aggregate>_s`, such as `sycl_avg_sum_s`.

LIKWID metric names in `objectives` must also be present in `likwid.metrics`.

## Outputs and failures

- `csv_log`: completed evaluation rows.
- `pareto_log`: current Pareto front for multi-objective Optuna studies.
- `failed_builds`: failed and pruned Optuna trials with their reason; also the tabu failure log.
- `sqlite_log`: persistent Optuna study database, written during optimization.

Parent directories are created automatically. For Optuna and wavefront, a CSV that already exists
triggers a timestamped filename instead of an overwrite. Tabu keeps the configured filename stable,
because the results CSV doubles as its resume record; see
[Resuming a tabu run](#resuming-a-tabu-run) for how earlier files are archived.

Temporary build artifacts and logs are printed at startup, normally under `/tmp/SCOuT_*`. For tabu
the location is stable across restarts (`$TMPDIR/SCOuT_tabu` unless overridden), which is what allows
previously built binaries to be reused while they are still on disk.
