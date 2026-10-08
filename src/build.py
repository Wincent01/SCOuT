from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
import optuna

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import BuildProject
from src.logger import LOG

###############################################################################
# Shell helpers                                                               #
###############################################################################

def _run(cmd: Sequence[str] | str, *, cwd: Path | None = None, env: EnvMap | None = None) -> subprocess.CompletedProcess:
    """Run a command, capturing output, and echo it to the console."""
    pretty = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(c)) for c in cmd)
    LOG.log(f"[exec] {pretty}" + (f"  (cwd={cwd})" if cwd else ""))
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

# Progress line patterns: Ninja prints "[N/M] …", CMake Makefiles prints "[ NN%] …".
_NINJA_RE = re.compile(r"^\[(\d+)\s*/\s*(\d+)\]")
_CMAKE_RE = re.compile(r"^\[\s*(\d{1,3})\s*%\]")


def _progress_pct(line: str) -> Optional[int]:
    """Return the integer progress percentage encoded in a build line, else None."""
    m = _NINJA_RE.match(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        return round(100 * done / total) if total else None
    m = _CMAKE_RE.match(line)
    if m:
        return int(m.group(1))
    return None


def _run_stream(cmd: Sequence[str] | str, *, cwd: Path | None = None, env: EnvMap | None = None) -> subprocess.CompletedProcess:
    """
    Run a command, streaming its output to the console with live progress
    updates, while capturing everything for later logging.

    stdout and stderr are merged (stderr=STDOUT). Progress is detected from
    Ninja ('[N/M]') or CMake ('[ NN%]') lines and echoed once per whole-percent
    step. Returns a CompletedProcess whose .stdout holds the merged output and
    whose .stderr is empty.
    """
    pretty = " ".join(shlex.quote(str(c)) for c in cmd) if isinstance(cmd, Sequence) else cmd
    LOG.log(f"[exec] {pretty}" + (f"  (cwd={cwd})" if cwd else ""))
    LOG.log("[build] starting …")

    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    lines: List[str] = []
    last_pct: Optional[int] = None
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        pct = _progress_pct(line)
        if pct is not None and pct != last_pct:
            last_pct = pct
            LOG.log(f"[build] {pct:3d}%  {line.strip()}")

    rc = proc.wait()
    LOG.log(f"[build] finished (rc={rc})")
    return subprocess.CompletedProcess(cmd, rc, stdout="".join(lines), stderr="")

def _trial_tag(trial: Optional["optuna.Trial"]) -> str:
    return f"trial_{trial.number:05d}" if trial is not None else f"phaseB_{uuid.uuid4().hex[:8]}"

def _save_log(workdir: Path,
              trial: Optional["optuna.Trial"],
              step: str,
              proc) -> Path:
    """
    Save stdout/stderr of a subprocess to workdir/logs and return the log dir.
    Works even when trial is None (e.g., Phase-B rebuilds).
    """
    log_dir = Path(workdir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = _trial_tag(trial)
    (log_dir / f"{tag}_{step}.out").write_text(proc.stdout or "")
    (log_dir / f"{tag}_{step}.err").write_text(proc.stderr or "")
    return log_dir


def _report_build_failure(step: str, proc, log_dir: Path) -> None:
    """Print a concise terminal report for a failed build step."""
    print(f"[build] ✗ {step} FAILED (rc={proc.returncode})")
    print(f"[build] logs → {log_dir}")
    text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    tail = text.strip().splitlines()[-20:]
    if tail:
        print(f"[build] {step} — last output:")
        for ln in tail:
            print(f"  {ln}")

###############################################################################
# Build logic (identical to original)                                         #
###############################################################################

def compile_single_source(compiler: str, src: Path, flags: str, out: Path, trial: Optional[optuna.Trial] = None) -> Optional[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*shlex.split(compiler), *shlex.split(flags), str(src), "-o", str(out)]
    proc = _run(cmd)
    if proc.returncode:
        log_dir = _save_log(out, trial, "compile", proc)
        _report_build_failure("compile", proc, log_dir)
        return None
    return out if out.is_file() else None


def _last_executable(root: Path) -> Optional[Path]:
    latest: Optional[Path] = None
    latest_mtime = -1.0
    for p in root.rglob("*"):
        if p.is_file() and os.access(p, os.X_OK):
            mtime = p.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime, latest = mtime, p
    return latest


def _resolve_cmake_target_executable(build_dir: Path, target: str) -> Optional[Path]:
    # Common CMake layout: executable under build root or build/bin.
    direct = build_dir / target
    if direct.is_file() and os.access(direct, os.X_OK):
        return direct

    in_bin = build_dir / "bin" / target
    if in_bin.is_file() and os.access(in_bin, os.X_OK):
        return in_bin

    # Fallback: search exact filename under the build tree.
    matches: List[Path] = []
    for p in build_dir.rglob(target):
        if p.is_file() and os.access(p, os.X_OK):
            matches.append(p)
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

def compile_project(cfg: BuildProject, compiler: str, flags: str, workdir: Path, trial: Optional[optuna.Trial] = None) -> Optional[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    if cfg.build_system == "cmake":
        build_dir = workdir / f"cmake_{uuid.uuid4().hex[:8]}"
        build_dir.mkdir()
        defs = cfg.cmake_defs

        cmake_cmd = [
            "cmake", "-S", str(cfg.dir), "-B", str(build_dir),
             f"-DCMAKE_CXX_COMPILER={compiler}",
             "-DCMAKE_BUILD_TYPE=Release"
            ] 
        
        if flags and cfg.cmake_flag_vars:
            for var in cfg.cmake_flag_vars:
                cmake_cmd += [f"-D{var}:STRING={flags}"]
        
        if cfg.cmake_defs:
            cmake_cmd +=[f"-D{d}" for d in defs]

        proc = _run_stream(cmake_cmd)
        if proc.returncode:
            log_dir = _save_log(workdir, trial, "cmake_config", proc)
            _report_build_failure("cmake_config", proc, log_dir)
            return None

        build_cmd = ["cmake", "--build", str(build_dir)]
        if cfg.build_jobs and cfg.build_jobs > 0:
            build_cmd += ["--parallel", str(cfg.build_jobs)]
        else:
            build_cmd += ["--parallel"]
        if cfg.target:
            build_cmd += ["--target", cfg.target]
        proc = _run_stream(build_cmd)
        if proc.returncode:
            log_dir = _save_log(workdir, trial, "cmake_build", proc)
            _report_build_failure("cmake_build", proc, log_dir)
            return None

        binary = _resolve_project_executable(cfg, build_dir)
        if binary is None:
            _save_message(workdir, trial, "cmake_artifact", "Build succeeded but no executable was found")
        return binary

    if cfg.build_system == "make":
        clean = _run(["make", "clean"], cwd=cfg.dir)
        if clean.returncode:
            _save_log(workdir, trial, "make_clean", clean)
            return None

        # Remove a previously built, explicitly declared target. The project's own
        # clean target may not remove it, and because compiler flags are passed as
        # make variables (not file changes), make could otherwise treat the old
        # binary as up to date and hand back a previous run's artifact.
        stale = _declared_executable_path(cfg, cfg.dir)
        if stale is not None and (stale.is_file() or stale.is_symlink()):
            try:
                stale.unlink()
            except OSError:
                pass

        # ``--always-make`` forces recompilation/relinking regardless of timestamps,
        # so the returned binary always reflects this configuration's flags.
        build_cmd = ["make", "--always-make", f"CXX={compiler}"]
        if cfg.build_jobs and cfg.build_jobs > 0:
            build_cmd.append(f"-j{cfg.build_jobs}")
        else:
            build_cmd.append("-j")

        if flags:
            build_cmd.append(f"{cfg.make_flags_var}+={flags}")
        for var, val in cfg.make_vars.items():
            build_cmd.append(f"{var}={val}")
        if cfg.target:
            build_cmd.append(cfg.target)

        proc = _run(build_cmd, cwd=cfg.dir)
        if proc.returncode:
            log_dir = _save_log(workdir, trial, "make", proc)
            _report_build_failure("make", proc, log_dir)
            return None
        
        binary = _resolve_project_executable(cfg, cfg.dir)
        if binary is None:
            _save_message(workdir, trial, "make_artifact", "Build succeeded but no executable was found")
        return binary

    raise ValueError(f"unknown build_system '{cfg.build_system}'")

def _declared_executable_path(cfg: BuildProject, build_root: Path) -> Optional[Path]:
    """Path of the explicitly configured executable/target, or None.

    Unlike :func:`_resolve_project_executable` this never searches the build tree,
    so it is safe to use for deleting a specific, configured artifact.
    """
    if cfg.executable:
        return cfg.executable if cfg.executable.is_absolute() else build_root / cfg.executable
    if cfg.target:
        return build_root / cfg.target
    return None


def _resolve_project_executable(cfg: BuildProject, build_root: Path) -> Optional[Path]:
    """Resolve the runnable artifact independently from the build target name."""
    if cfg.executable:
        candidate = cfg.executable if cfg.executable.is_absolute() else build_root / cfg.executable
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None

    if cfg.target:
        candidate = build_root / cfg.target
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return _last_executable(build_root)


def _save_message(workdir: Path, trial: Optional["optuna.Trial"], step: str, message: str) -> None:
    class _Message:
        stdout = ""
        stderr = message

    _save_log(workdir, trial, step, _Message())
