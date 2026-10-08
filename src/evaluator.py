from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from src.build import compile_project, compile_single_source
from src.checkpoint import env_key
from src.config import Config
from src.metrics import measure_likwid, measure_parser_sycl, measure_perf
from src.stats import STATS


MetricDict = Dict[str, float]


def _is_within(path: Path, root: Path) -> bool:
    """True when ``path`` resolves to a location inside ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


@dataclass(frozen=True)
class Evaluation:
    metrics: MetricDict
    binary: Path
    cached: bool = False


class Evaluator:
    """
    Shared build and measurement service with separate build/evaluation caches.

    Builds live under ``workroot/builds/<compile-key>``. When that directory is
    still present from an earlier process (for example after a resubmitted job
    that reuses the same node-local workroot), the existing artifact is reused
    instead of rebuilt; otherwise it is simply built again.
    """

    META_NAME = "build_meta.json"

    def __init__(self, cfg: Config, workroot: Path, *, cache_evaluations: bool = True):
        self.cfg = cfg
        self.workroot = Path(workroot)
        self.workroot.mkdir(parents=True, exist_ok=True)
        self.cache_evaluations = cache_evaluations
        self._build_cache: Dict[str, Path] = {}
        self._build_failures: Dict[str, str] = {}
        self._evaluation_cache: Dict[tuple[str, tuple[tuple[str, str], ...]], Evaluation] = {}

    @staticmethod
    def normalize_flags(flags: str | Sequence[str]) -> str:
        if isinstance(flags, str):
            return shlex.join(shlex.split(flags)) if flags.strip() else ""
        tokens: list[str] = []
        for fragment in flags:
            tokens.extend(shlex.split(str(fragment)))
        return shlex.join(tokens) if tokens else ""

    def compile_key(self, flags: str | Sequence[str]) -> str:
        normalized = self.normalize_flags(flags)
        project = self.cfg.project
        payload = {
            "compiler": self.cfg.compiler,
            "flags": normalized,
            "source": str(self.cfg.source) if self.cfg.source else None,
            "project": (
                {
                    "dir": str(project.dir),
                    "build_system": project.build_system,
                    "target": project.target,
                    "executable": str(project.executable) if project.executable else None,
                    "make_vars": project.make_vars,
                    "make_flags_var": project.make_flags_var,
                    "cmake_defs": project.cmake_defs,
                    "cmake_flag_vars": project.cmake_flag_vars,
                }
                if project
                else None
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # Persistent build reuse
    def expected_artifact_name(self) -> Optional[str]:
        """Name of the artifact that ``get_or_build`` produces, if known."""
        if self.cfg.source:
            return "program"
        project = self.cfg.project
        if project and project.executable:
            return Path(project.executable).name
        return None

    def build_dir_for(self, compile_key: str) -> Path:
        return self.workroot / "builds" / compile_key[:16]

    def run_dir_for(self, flags: str | Sequence[str], env: Dict[str, str]) -> Path:
        """
        Deterministic scratch directory for measuring one configuration.

        Stable across restarts so that leftover logs and outputs of an
        interrupted run are found again and can be cleaned before re-measuring.
        """
        digest = hashlib.sha256(env_key(env).encode("utf-8")).hexdigest()[:16]
        return self.workroot / "runs" / self.compile_key(flags)[:16] / digest

    def _read_build_meta(self, build_dir: Path) -> Optional[dict]:
        meta_path = build_dir / self.META_NAME
        if not meta_path.is_file():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_build_meta(self, build_dir: Path, key: str, flags: str, artifact: Path) -> None:
        payload = {
            "compile_key": key,
            "flags": flags,
            "artifact": artifact.name,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            file = (build_dir / self.META_NAME)
            file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass

    def probe_artifact(self, compile_key: str, build_dir: Path | None = None) -> Optional[Path]:
        """
        Return an existing, usable artifact for ``compile_key`` or None otherwise.

        Nothing is assumed to still be on disk. The build metadata must match
        this key and the artifact must be executable.
        """
        build_dir = Path(build_dir) if build_dir is not None else self.build_dir_for(compile_key)
        meta = self._read_build_meta(build_dir)
        if not meta or meta.get("compile_key") != compile_key:
            return None
        name = meta.get("artifact") or self.expected_artifact_name()
        if not name:
            return None
        candidate = build_dir / "artifact" / str(name)
        if candidate.is_file() and os.access(candidate, os.X_OK): # Check if the file is executable
            return candidate
        return None

    def seed_evaluation(
        self,
        flags: str | Sequence[str],
        env: Dict[str, str],
        metrics: Dict[str, float],
        binary: Union[str, Path, None] = None,
    ) -> bool:
        """
        Register a result measured by an earlier process.

        Returns True when the entry was added, so callers can report how much
        prior work is being reused.
        """
        compile_key = self.compile_key(flags)
        cache_key = (compile_key, tuple(sorted((str(k), str(v)) for k, v in (env or {}).items())))
        if cache_key in self._evaluation_cache:
            return False
        clean: Dict[str, float] = {}
        for name, value in (metrics or {}).items():
            try:
                clean[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        if not clean:
            return False
        artifact = self.probe_artifact(compile_key)
        binary_path = Path(binary) if binary else (artifact or self.workroot)
        self._evaluation_cache[cache_key] = Evaluation(clean, binary_path, True)
        if artifact is not None:
            self._build_cache[compile_key] = artifact
        self._build_failures.pop(compile_key, None)
        return True

    def seed_failure(self, flags: str | Sequence[str], reason: str) -> None:
        """Remember a failure recorded by an earlier process."""
        key = self.compile_key(flags)
        self._build_failures.setdefault(key, f"previous attempt failed: {reason}")
        self._build_cache.pop(key, None)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def get_or_build(self, flags: str | Sequence[str]) -> tuple[str, Path]:
        key = self.compile_key(flags)
        cached = self._build_cache.get(key)
        if cached is not None and cached.is_file():
            return key, cached

        # Check for presitent artifact on disk before building.
        build_dir = self.build_dir_for(key)
        reused = self.probe_artifact(key, build_dir)
        if reused is not None:
            self._build_cache[key] = reused
            self._build_failures.pop(key, None)
            return key, reused

        if key in self._build_failures:
            raise RuntimeError(self._build_failures[key])

        meta = self._read_build_meta(build_dir)
        if meta is not None and meta.get("compile_key") != key:
            # Old tree under this key: rebuild from scratch instead of
            # inheriting a mismatched incremental build.
            shutil.rmtree(build_dir, ignore_errors=True)
            # Drop in-memory entries pointing into the removed tree so a later
            # request cannot be served a path that no longer belongs to its key.
            for cached_key, cached_path in list(self._build_cache.items()):
                if _is_within(cached_path, build_dir):
                    del self._build_cache[cached_key]

        build_dir.mkdir(parents=True, exist_ok=True)
        normalized = self.normalize_flags(flags)

        # normalize_flags uses shell quoting for stable keys; compiler APIs expect a
        # normal shell fragment and tokenize it safely themselves.
        flags_str = " ".join(shlex.quote(token) for token in shlex.split(normalized))
        with STATS.time_build():
            if self.cfg.source:
                binary = compile_single_source(self.cfg.compiler, self.cfg.source, flags_str, build_dir / "program")
            elif self.cfg.project:
                binary = compile_project(self.cfg.project, self.cfg.compiler, flags_str, build_dir)
            else:  # Config validation should make this unreachable.
                raise RuntimeError("no source or project configured")
        if not binary:
            message = f"build failed; logs are under {build_dir / 'logs'}"
            self._build_failures[key] = message
            raise RuntimeError(message)

        binary = Path(binary)
        artifact = build_dir / "artifact" / binary.name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        # gmx builds break if the binary is copied to a different path, so we only copy it if it's not gmx.
        # gmx attempts to load libraries relative to the binary path, so copying it breaks that.
        if binary.resolve() != artifact.resolve() and ("gmx" not in binary.name):
            shutil.copy2(binary, artifact)
            artifact.chmod(artifact.stat().st_mode | 0o111)
        else:
            artifact = binary

        # Only artifacts that live inside this key's build directory may be cached
        # or persisted. A raw build output elsewhere (for example a "gmx" binary
        # produced by make, or an absolute project.executable) is shared between
        # configurations: a later build would overwrite it, so serving it from the
        # in-memory build cache would hand one configuration another's binary.
        if _is_within(artifact, build_dir):
            self._write_build_meta(build_dir, key, flags_str, artifact)  # allow future reuse
            self._build_cache[key] = artifact
        return key, artifact

    def evaluate(self, flags: str | Sequence[str], env: Dict[str, str], run_workdir: Path) -> Evaluation:
        compile_key, binary = self.get_or_build(flags)
        env_key = tuple(sorted((str(k), str(v)) for k, v in env.items()))
        key = (compile_key, env_key)
        cached = self._evaluation_cache.get(key)
        if self.cache_evaluations and cached is not None:
            return Evaluation(dict(cached.metrics), cached.binary, True)

        run_workdir = Path(run_workdir)
        run_workdir.mkdir(parents=True, exist_ok=True)
        managed_keys = tuple((self.cfg.env or {}).keys())
        clear_runtime = self.cfg.runtime_cache_policy == "cold"

        with STATS.time_run():
            if self.cfg.backend == "perf":
                metrics = measure_perf(
                    self.cfg.perf, binary, self.cfg.program_args, env, self.cfg.runs,
                    clear_runtime_cache=clear_runtime, managed_env_keys=managed_keys,
                )
            elif self.cfg.backend == "parser":
                metrics = measure_parser_sycl(
                    self.cfg.parser, binary, self.cfg.program_args, env, self.cfg.runs,
                    run_workdir, self.cfg.project,
                    clear_runtime_cache=clear_runtime, managed_env_keys=managed_keys,
                )
            else:
                metrics = measure_likwid(
                    self.cfg.likwid, binary, self.cfg.program_args, env, self.cfg.runs,
                    clear_runtime_cache=clear_runtime, managed_env_keys=managed_keys,
                )

        result = Evaluation({str(k): float(v) for k, v in metrics.items()}, binary)
        if self.cache_evaluations:
            self._evaluation_cache[key] = result
        return result

    @property
    def build_count(self) -> int:
        return len(self._build_cache)
