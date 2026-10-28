from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import dataclasses
import os
import re
import sys, subprocess
from pathlib import Path
from statistics import mean, median, stdev, variance
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import PerfConfig, MetricSpec, LikwidConfig, ParserConfig, BuildProject
from src.build import _run
from src.misc import clear_acpp_runtime_cache

###############################################################################
# Measurement helpers (perf & likwid)                                         #
###############################################################################

_PERF_LINE_RE = re.compile(r"^\s*([0-9,]+)\s+([^\s#]+)")


def _measurement_env(env: EnvMap, managed_env_keys: Sequence[str] = ()) -> EnvMap:
    """Overlay a candidate environment after removing all managed search keys."""
    merged = dict(os.environ)
    for key in managed_env_keys:
        merged.pop(str(key), None)
    merged.update({str(k): str(v) for k, v in env.items()})
    return merged


def _mean_and_stdev(buckets: Mapping[str, Sequence[Number]]) -> MetricDict:
    """
    Reduce per-run metric buckets to their mean and, when more than one sample
    is available, the sample standard deviation published as ``<metric>_stdev``.

    A single sample has no defined sample standard deviation, so the
    ``_stdev`` key is omitted in that case.
    """
    summary: MetricDict = {}
    for key, values in buckets.items():
        if not values:
            continue
        summary[str(key)] = mean(values)
        if len(values) > 1:
            summary[f"{key}_stdev"] = stdev(values)
    return summary


def perf_parse(stderr: str, events: Sequence[str]) -> MetricDict:
    accum: Dict[str, Number] = {}
    for line in stderr.splitlines():
        match = _PERF_LINE_RE.match(line)
        if not match:
            continue
        raw_value, raw_event = match.groups()
        try:
            count = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        base_event = raw_event.strip("/").split("/")[-1]
        if base_event not in events:
            continue
        accum[base_event] = accum.get(base_event, 0.0) + count
    if {"cycles", "instructions"}.issubset(accum) and accum["instructions"]:
        accum["CPI"] = accum["cycles"] / accum["instructions"]
    return accum

def measure_perf(
    cfg: PerfConfig,
    bin_path: Path,
    prog_args: List[str],
    env: EnvMap,
    runs: int = 1,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
    *,
    clear_runtime_cache: bool = True,
    managed_env_keys: Sequence[str] = (),
) -> MetricDict:
    meas_runs = max(1, runs)
    total_runs = cfg.warmup_runs + meas_runs
    buckets: Dict[str, List[Number]] = {e: [] for e in cfg.events + ["CPI"]}
    cwd = resolve_run_cwd(getattr(cfg, "run_cwd", "process"), bin_path, workdir, project)

    for i in range(total_runs):
        cmd = ["perf", "stat", "-e", ",".join(cfg.events), "--", str(bin_path), *prog_args]
        if cfg.core_list:
            cmd = ["taskset", "-c", cfg.core_list, *cmd]
        proc = _run(cmd, cwd=cwd, env=_measurement_env(env, managed_env_keys))
        if proc.returncode != 0:
            raise RuntimeError(f"perf/program exited with rc={proc.returncode}: {(proc.stderr or '')[-1200:]}")

        # Parse always, but only store if not warm-up
        data = perf_parse(proc.stderr, cfg.events)
        if i < cfg.warmup_runs:
            continue
        if not data:
            raise RuntimeError("Perf parse failure – received no matching events.")
        for k, v_list in buckets.items():
            if k in data:
                v_list.append(data[k])
    
    if clear_runtime_cache:
        clear_acpp_runtime_cache()
    return _mean_and_stdev(buckets)


_ROW_RE   = re.compile(r"^\|\s*([^|]+?)\s*\|(.+)$")
_SEP_RE   = re.compile(r"(?<=\d)[.'\u202F](?=\d{3}\b)")  # 1.234.567 or 1'234'567
_DEC_COMMA = re.compile(r"^(\d+),(\d+)$")  
_LIKWID_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|(.+)$")

def _parse_num(text: str) -> float | None:
    """Parse tolerant float or return None."""
    t = text.strip()
    t = _SEP_RE.sub("", t)              # 1.234.567 -> 1234567
    m = _DEC_COMMA.match(t)             # decimal comma?
    if m:
        t = f"{m.group(1)}.{m.group(2)}"
    try:
        return float(t)
    except ValueError:
        return None

def likwid_parse(out: str, specs: Sequence[MetricSpec]) -> MetricDict:
    wanted = {s.name: s for s in specs}

    # Prepare buckets
    per_thread: Dict[str, List[Number]] = {s.name: [] for s in specs}
    stat_avg:   Dict[str, Number]       = {}

    # ── scan once ────────────────────────────────────────────────────
    for line in out.splitlines():
        m = _ROW_RE.match(line)
        if not m:                   # skip non-table lines
            continue
        name, cells_raw = m.group(1).strip(), m.group(2)

        # 1) STAT rows  → grab Avg column (index 3)
        if name.endswith("STAT"):
            base = name[:-4].rstrip()
            if base in wanted:
                cells = [c.strip() for c in cells_raw.split("|") if c.strip()]
                if len(cells) >= 4:
                    v = _parse_num(cells[3])
                    if v is not None:
                        stat_avg[base] = v
            continue

        # 2) per-thread rows
        if name in wanted:
            for cell in (c for c in cells_raw.split("|") if c.strip()):
                v = _parse_num(cell)
                if v is not None:
                    per_thread[name].append(v)

    # ── reduce according to spec ─────────────────────────────────────
    result: MetricDict = {}
    for spec in specs:
        values = per_thread[spec.name]

        # if we have STAT Avg and user asked for avg—use it (cheaper)
        if spec.agg == "avg" and spec.name in stat_avg:
            agg_val = stat_avg[spec.name]
        elif values:
            if   spec.agg == "avg": agg_val = mean(values)
            elif spec.agg == "max": agg_val = max(values)
            elif spec.agg == "min": agg_val = min(values)
            elif spec.agg == "median": agg_val = median(values)
            else:
                raise ValueError(f"Unknown agg mode '{spec.agg}'")
        else:
            continue  # metric missing

        result[spec.name] = agg_val

        if spec.var and len(values) > 1:
            result[f"{spec.name}_var"] = variance(values)

    return result

def measure_likwid(
    cfg: LikwidConfig,
    bin_path: Path,
    prog_args: List[str],
    env: EnvMap,
    runs: int = 1,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
    *,
    clear_runtime_cache: bool = True,
    managed_env_keys: Sequence[str] = (),
) -> MetricDict:
    specs   = cfg.metrics
    buckets: Dict[str, List[Number]] = {s.name: [] for s in specs}
    for s in specs:
        if s.var:
            buckets[f"{s.name}_var"] = []
    
    meas_runs = max(1, runs)
    total_runs = cfg.warmup_runs + meas_runs
    cwd = resolve_run_cwd(getattr(cfg, "run_cwd", "process"), bin_path, workdir, project)

    for i in range(total_runs):
        cmd = ["likwid-perfctr"]

        if cfg.core_list:
            cmd += ["-C", cfg.core_list]

        if cfg.group:
            cmd += ["-g", cfg.group]
        else:                       # raw events
            cmd += ["-g", ",".join(cfg.events)]
        cmd += [str(bin_path), *prog_args]

        proc = _run(cmd, cwd=cwd, env=_measurement_env(env, managed_env_keys))
        if proc.returncode != 0:
            text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            raise RuntimeError(f"LIKWID/program exited with rc={proc.returncode}: {text[-1200:]}")
        data = likwid_parse(proc.stdout, cfg.metrics)

        if i < cfg.warmup_runs:
            continue
        if not data:
            raise RuntimeError("LIKWID parse failure – no metrics captured.")
        for k, v in data.items():
            if k in buckets:
                buckets[k].append(v)

    if clear_runtime_cache:
        clear_acpp_runtime_cache()
    return _mean_and_stdev(buckets)


###############################################################################
# Parser helpers (Parsers for HeCBench)                                       #
###############################################################################

# Matches: [SYCL][avg] kernel 2: 0.000664 s over 1000 iters
_NUMBER_RE = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_SYCL_RE = re.compile(
    r'^\[SYCL\]\[(?P<label>avg|sum)\]\s*kernel\s*(?P<kid>\d+)\s*:\s*'
    rf'(?P<val>{_NUMBER_RE})\s*s\s*over\s*(?P<iters>\d+)\s*iters\s*$',
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

_GMX_DEVICE_LINE_RE = re.compile(r"^\s*#(?P<id>\d+):\s*(?P<desc>.*\S)\s*$")
_GMX_GPU_MAPPING_RE = re.compile(r"(?P<task>PP|PME):(?P<id>\d+)")


def _gromacs_log_path(prog_args: Sequence[str], cwd: Path) -> Path:
    """Locate the mdrun .log file from -g / -deffnm (default md.log)."""
    args = list(prog_args)
    for opt, suffix in (("-g", ""), ("-deffnm", ".log")):
        if opt in args and args.index(opt) + 1 < len(args):
            return cwd / (args[args.index(opt) + 1] + suffix)
    return cwd / "md.log"


def report_gromacs_device(cfg: ParserConfig, prog_args: Sequence[str], cwd: Path, logs: Path,
                          stdout: str, stderr: str) -> None:
    """
    Print the SYCL devices mdrun detected and the device each GPU task was mapped to,
    and keep the raw run output (e.g. ACPP_DEBUG_LEVEL messages) under ``logs``.
    Raises if ``cfg.expect_device`` is set and the selected device does not match.
    """
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "device_run.out").write_text(stdout or "")
    (logs / "device_run.err").write_text(stderr or "")

    md_log = _gromacs_log_path(prog_args, cwd)
    text = md_log.read_text(errors="replace") if md_log.exists() else ""

    # "GPU info:" block of the hardware report: one "#<id>: name: ..., status: ..." line per device.
    devices: Dict[int, str] = {}
    in_gpu_info = False
    for line in text.splitlines():
        if line.strip() == "GPU info:":
            in_gpu_info = True
            continue
        if in_gpu_info:
            m = _GMX_DEVICE_LINE_RE.match(line)
            if m:
                devices[int(m.group("id"))] = m.group("desc")
            elif not line.strip() or not line.startswith("    "):
                in_gpu_info = False

    # Task assignment, e.g. "Mapping of GPU IDs to the 2 GPU tasks ...:\n  PP:0,PME:0"
    lines = text.splitlines()
    assignment: List[Tuple[str, int]] = []
    usage: List[str] = []
    for idx, line in enumerate(lines):
        if line.startswith("Mapping of GPU IDs") and idx + 1 < len(lines):
            assignment = [(m.group("task"), int(m.group("id")))
                          for m in _GMX_GPU_MAPPING_RE.finditer(lines[idx + 1])]
        if re.match(r"^(\d+ GPUs? selected|PP tasks? will|PME tasks? will)", line):
            usage.append(line.strip())

    tag = "[scout][gmx-device]"
    print(f"{tag} mdrun log: {md_log}{'' if text else ' (not found)'}", flush=True)
    print(f"{tag} raw output of this run saved under {logs}/device_run.{{out,err}}", flush=True)
    if not devices:
        print(f"{tag} no 'GPU info' devices in log (no SYCL devices detected/visible)", flush=True)
    for dev_id, desc in sorted(devices.items()):
        print(f"{tag} detected #{dev_id}: {desc}", flush=True)
    for line in usage:
        print(f"{tag} {line}", flush=True)
    if not assignment:
        print(f"{tag} no GPU tasks assigned: all work ran on the host CPU (native SIMD), "
              f"not through SYCL", flush=True)
    for task, dev_id in assignment:
        print(f"{tag} {task} task -> #{dev_id}: {devices.get(dev_id, '<unknown device>')}", flush=True)

    if cfg.expect_device:
        if not assignment:
            raise RuntimeError(f"expect_device={cfg.expect_device!r} but mdrun assigned no GPU tasks "
                               f"(see {md_log})")
        for task, dev_id in assignment:
            desc = devices.get(dev_id, "")
            if not re.search(cfg.expect_device, desc):
                raise RuntimeError(f"{task} task ran on device #{dev_id} ({desc or 'unknown'}), "
                                   f"which does not match expect_device={cfg.expect_device!r}")

def resolve_run_cwd(
    run_cwd: Optional[str],
    bin_path: Path,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
) -> Optional[Path]:
    """
    Working directory for a measured program.

    ``process``/``cwd`` (the perf/likwid default) returns ``None`` so the child
    inherits SCOuT's working directory, preserving the historical behaviour.
    ``binary_dir`` returns the directory holding the executable (the parser
    backend's default), ``project_dir`` the configured project directory and
    ``workdir`` the per-evaluation scratch directory. An unknown value falls
    back to ``binary_dir``.
    """
    mode = str(run_cwd if run_cwd is not None else "process").strip().lower()
    if mode in {"", "process", "cwd", "inherit"}:
        return None
    if mode == "workdir":
        return Path(workdir) if workdir is not None else Path(bin_path).parent
    if mode == "project_dir":
        return project.resolved_dir() if project is not None else Path(bin_path).parent
    # "binary_dir" and anything unrecognised
    return Path(bin_path).parent


def _resolve_cwd(run_cwd: str, bin_path: Path, workdir: Optional[Path], project: Optional[BuildProject]) -> Path:
    """Parser resolver: always returns a concrete directory (never ``None``)."""
    resolved = resolve_run_cwd(run_cwd, bin_path, workdir, project)
    return resolved if resolved is not None else Path(bin_path).parent

def _aggregate(vals: List[float], how: str) -> float:
    how = how.lower()
    if how == "sum":  return float(sum(vals))
    if how == "mean": return float(mean(vals))
    if how == "max":  return float(max(vals))
    if how == "min":  return float(min(vals))
    return float(sum(vals))

def measure_parser_sycl(
    cfg: ParserConfig,
    bin_path: Path,
    prog_args: List[str],
    env: EnvMap,
    runs: int,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
    *,
    clear_runtime_cache: bool = True,
    managed_env_keys: Sequence[str] = (),
) -> MetricDict:
    """
    Runs the binary like perf/likwid (taskset/prefix, controlled cwd),
    parses lines printed as:
        [SYCL][avg] kernel K: <seconds> s over N iters
        [SYCL][sum] kernel K: <seconds> s over N iters
    Returns per-kernel metrics (seconds) and one aggregate key:
        sycl_<label>_<aggregate>_s
    """
    merged_env = _measurement_env(env, managed_env_keys)
    cmd: List[str] = []
    if cfg.prefix:     cmd.extend(cfg.prefix)
    if cfg.core_list:  cmd.extend(["taskset", "-c", cfg.core_list])
    cmd.append(str(bin_path))
    cmd.extend(prog_args)
    cwd = _resolve_cwd(cfg.run_cwd, bin_path, workdir, project)

    # collect per-run kernel maps {kid -> value_s}
    runs_kernel_vals: List[Dict[int, float]] = []
    iterations_seen: List[int] = []

    meas_runs = max(1, runs)
    total_runs = cfg.warmup_runs + meas_runs

    for i in range(total_runs):
        proc = _run(cmd, cwd=cwd, env=merged_env)
        if proc.returncode != 0:
            logs = (workdir or cwd) / "parser_logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"rc_{proc.returncode}_{i:02d}.out").write_text(proc.stdout or "")
            (logs / f"rc_{proc.returncode}_{i:02d}.err").write_text(proc.stderr or "")
            text = ((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")).strip()
            excerpt = text[-1200:] if text else "<no stdout/stderr>"
            if (cfg.format or "").lower() == "gromacs" and (cfg.device_report or cfg.expect_device):
                # mdrun logs device detection and GPU task mapping before launching kernels,
                # so the report is still informative for a crashed run.
                try:
                    report_gromacs_device(dataclasses.replace(cfg, expect_device=None), prog_args, cwd,
                                          logs, proc.stdout or "", proc.stderr or "")
                except Exception as exc:
                    print(f"[scout][gmx-device] report failed: {exc}", flush=True)
            raise RuntimeError(
                f"program exited with rc={proc.returncode}; "
                f"logs saved under {logs}; output excerpt:\n{excerpt}"
            )

        # For warm-up iterations, don’t enforce parsing or errors
        is_warmup = i < cfg.warmup_runs
        text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")

        parse_format = (getattr(cfg, "format", "sycl") or "sycl").lower()
        if parse_format == "gromacs":
            if i == 0 and (cfg.device_report or cfg.expect_device):
                report_gromacs_device(cfg, prog_args, cwd, (workdir or cwd) / "parser_logs",
                                      proc.stdout or "", proc.stderr or "")
            # Discard warm-up runs so the average covers exactly `runs` measurements.
            if is_warmup:
                continue
            perf_matches = list(_GROMACS_PERF_RE.finditer(text))
            if not perf_matches:
                logs = (workdir or cwd) / "parser_logs"
                logs.mkdir(parents=True, exist_ok=True)
                (logs / f"no_match_{i:02d}.out").write_text(proc.stdout or "")
                (logs / f"no_match_{i:02d}.err").write_text(proc.stderr or "")
                raise RuntimeError("Parser backend (GROMACS): no matching Performance line found.")
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
            if label != cfg.label.lower():
                continue
            kid   = int(m.group("kid"))
            val   = float(m.group("val"))       # seconds
            iters = int(m.group("iters"))

            iters_val = iters
            per_kernel[kid] = val

        if is_warmup:
            continue

        if not per_kernel:
        
            logs = (workdir or cwd) / "parser_logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"no_match_{i:02d}.out").write_text(proc.stdout or "")
            (logs / f"no_match_{i:02d}.err").write_text(proc.stderr or "")
            raise RuntimeError("Parser backend (SYCL): no matching [SYCL] lines found.")

        runs_kernel_vals.append(per_kernel)
        iterations_seen.append(iters_val if iters_val is not None else -1)

    parse_format = (getattr(cfg, "format", "sycl") or "sycl").lower()
    if parse_format == "gromacs":
        ns_per_day_vals = [d[0] for d in runs_kernel_vals if 0 in d]
        hour_per_ns_vals = [d[1] for d in runs_kernel_vals if 1 in d]
        ms_per_step_vals = [d[2] for d in runs_kernel_vals if 2 in d]
        matom_steps_vals = [d[3] for d in runs_kernel_vals if 3 in d]
        if not ns_per_day_vals:
            raise RuntimeError("Parser backend (GROMACS): no throughput values collected.")

        mets: MetricDict = {
            "gmx_ns_per_day": float(mean(ns_per_day_vals)),
            "gmx_hour_per_ns": float(mean(hour_per_ns_vals)) if hour_per_ns_vals else 0.0,
            "gmx_ms_per_step": float(mean(ms_per_step_vals)) if ms_per_step_vals else 0.0,
            "gmx_matom_steps_per_s": float(mean(matom_steps_vals)) if matom_steps_vals else 0.0,
        }
        # Sample standard deviation across the measurement repetitions.
        for key, values in (
            ("gmx_ns_per_day", ns_per_day_vals),
            ("gmx_hour_per_ns", hour_per_ns_vals),
            ("gmx_ms_per_step", ms_per_step_vals),
            ("gmx_matom_steps_per_s", matom_steps_vals),
        ):
            if len(values) > 1:
                mets[f"{key}_stdev"] = float(stdev(values))
        if clear_runtime_cache:
            clear_acpp_runtime_cache()
        return mets

    # average across runs per kernel
    all_kids = sorted({k for d in runs_kernel_vals for k in d.keys()})
    # filter by cfg.kernels if provided
    if cfg.kernels:
        selected = [k for k in all_kids if k in set(cfg.kernels)]
    else:
        selected = all_kids

    per_kernel_avg: Dict[int, float] = {}
    per_kernel_stdev: Dict[int, float] = {}
    for k in selected:
        vals = [d[k] for d in runs_kernel_vals if k in d]
        if not vals:
            continue
        per_kernel_avg[k] = float(mean(vals))
        if len(vals) > 1:
            per_kernel_stdev[k] = float(stdev(vals))

    if not per_kernel_avg:
        raise RuntimeError("Parser backend (SYCL): selected kernels missing in output.")

    # aggregate across selected kernels
    aggregate_val = _aggregate(list(per_kernel_avg.values()), cfg.aggregate)

    # Aggregate each run over the selected kernels present in it, so the stddev of
    # the aggregate reflects run-to-run variation rather than kernel composition.
    per_run_aggregate = [
        _aggregate([d[k] for k in selected if k in d], cfg.aggregate)
        for d in runs_kernel_vals
        if any(k in d for k in selected)
    ]

    # build metrics dict
    mets: MetricDict = {}
    # per-kernel seconds (mean + stddev across runs)
    for k, v in per_kernel_avg.items():
        mets[f"sycl_kernel_{k}_{cfg.label}_s"] = v
        if k in per_kernel_stdev:
            mets[f"sycl_kernel_{k}_{cfg.label}_s_stdev"] = per_kernel_stdev[k]
    # aggregate seconds
    mets[f"sycl_{cfg.label}_{cfg.aggregate}_s"] = aggregate_val
    if len(per_run_aggregate) > 1:
        mets[f"sycl_{cfg.label}_{cfg.aggregate}_s_stdev"] = float(stdev(per_run_aggregate))

    # if iterations consistent and positive, report it too
    if all(i == iterations_seen[0] and i >= 0 for i in iterations_seen):
        mets["sycl_iters"] = float(iterations_seen[0])

    if clear_runtime_cache:
        clear_acpp_runtime_cache()
    return mets
