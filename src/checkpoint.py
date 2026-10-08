from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import csv
import hashlib
import io
import json
import os
import shutil
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
MetricDict = Dict[str, Number]

STATE_SCHEMA = 1

# Columns of the tabu results CSV that are not objective/extra metrics.
RESULTS_FIXED_COLUMNS: Tuple[str, ...] = (
    "k",
    "compiler_flags",
    "env",
    "binary",
    "eval_id",
    "flags",
    "compile_key",
    "cached",
)

# Columns of the tabu failure CSV.
FAILURE_COLUMNS: Tuple[str, ...] = (
    "k",
    "eval_id",
    "flags",
    "env",
    "reason",
    "binary"
)


class StopRequested(BaseException):
    """
    Raised when an external stop signal (e.g. SLURM walltime) was received.
    """
    pass


_STOP = {"installed": False, "previous": None}


def install_sigterm_handler() -> None:
    """
    Install SIGTERM into StopRequested so that a walltime kill unwinds the
    search normally and lets it persist its state instead of dying silently.
    """
    if not hasattr(signal, "SIGTERM"):
        return

    def _handle(signum, _frame):
        raise StopRequested(f"received signal {signum}")

    try:
        _STOP["previous"] = signal.signal(signal.SIGTERM, _handle)
        _STOP["installed"] = True
    except (ValueError, OSError, RuntimeError):
        pass


def restore_sigterm_handler() -> None:
    """Restore the SIGTERM handler that was active before"""
    if not _STOP["installed"]:
        return
    try:
        signal.signal(signal.SIGTERM, _STOP["previous"])
    except (ValueError, OSError, RuntimeError):
        pass
    _STOP["installed"] = False


# Identity helpers


def _sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def env_key(env: Optional[Mapping[str, Any]]) -> str:
    """Canonical string form of an environment mapping in stable order."""
    return json.dumps({str(k): str(v) for k, v in sorted((env or {}).items())}, sort_keys=True)


def eval_id(compile_key: str, env: Optional[Mapping[str, Any]]) -> str:
    """Stable identifier of one evaluated configuration."""
    return _sha256({"compile_key": compile_key, "env": env_key(env)})[:24]


def config_fingerprint(cfg: Any, tabu_spec: Any) -> str:
    """
    Fingerprint of everything that defines the search space.

    Resume is refused when this changes, because continuing a trajectory in a
    different space may produce misleading results.
    """
    project = getattr(cfg, "project", None)
    payload = {
        "schema": 1,
        "backend": getattr(cfg, "backend", None),
        "compiler": getattr(cfg, "compiler", None),
        "compiler_flags_base": getattr(cfg, "compiler_flags_base", None),
        "compiler_flags": list(getattr(cfg, "compiler_flags", None) or []),
        "compiler_params": getattr(cfg, "compiler_params", None) or {},
        "compiler_params_select": getattr(cfg, "compiler_params_select", None) or {},
        "compiler_flag_pool": list(getattr(cfg, "compiler_flag_pool", None) or []),
        "env": getattr(cfg, "env", None) or {},
        "program_args": list(getattr(cfg, "program_args", None) or []),
        "runs": getattr(cfg, "runs", None),
        "objectives": [[o.metric, o.goal] for o in getattr(cfg, "objectives", None) or []],
        "source": str(cfg.source) if getattr(cfg, "source", None) else None,
        "project": None
        if project is None
        else {
            "dir": str(project.dir),
            "build_system": project.build_system,
            "target": project.target,
            "executable": str(project.executable) if project.executable else None,
            "make_vars": project.make_vars,
            "make_flags_var": project.make_flags_var,
            "cmake_defs": list(project.cmake_defs),
            "cmake_flag_vars": list(project.cmake_flag_vars),
        },
        "tabu_space": {
            "allow_variant_moves": getattr(tabu_spec, "allow_variant_moves", None),
            "allow_param_moves": getattr(tabu_spec, "allow_param_moves", None),
            "allow_pool_moves": getattr(tabu_spec, "allow_pool_moves", None),
            "allow_env_moves": getattr(tabu_spec, "allow_env_moves", None),
            "env_mode": getattr(tabu_spec, "env_mode", None),
            "env_cap": getattr(tabu_spec, "env_cap", None),
            "env": getattr(tabu_spec, "env", None) or {},
        },
    }
    # Only include flag constraints when present, so constraint-free configs keep
    # their existing fingerprints and stay resumable after this change.
    constraints = getattr(cfg, "flag_constraints", None)
    if constraints is not None and not getattr(constraints, "is_empty", True):
        payload["flag_constraints"] = constraints.as_fingerprint()
    return _sha256(payload)[:16]

# RNG helpers

def rng_to_json(state: Sequence[Any]) -> List[Any]:
    """Make random.Random.getstate a JSON serializable."""
    return [int(state[0]), [int(x) for x in state[1]], state[2]]


def rng_from_json(state: Sequence[Any]) -> Tuple[int, Tuple[int, ...], Any]:
    """Inverse rng_to_json"""
    return (int(state[0]), tuple(int(x) for x in state[1]), state[2])


@dataclass
class EvalRecord:
    """One successfully measured configuration, as persisted in the results CSV."""

    k: int
    eval_id: str
    value: Number
    flags_key: str
    flags: str
    env: Dict[str, str]
    binary: str
    compile_key: str
    metrics: MetricDict
    cached: bool = False

class CsvAppender:
    """
    Append-only CSV writer that keeps the header in sync with the columns written.
    """

    def __init__(self, path: Path | str, columns: Sequence[str], *, sync: bool = True):
        self.path = Path(path)
        self.columns = list(columns)
        self.sync = sync
        self._header: List[str] = []
        self._extras: List[str] = []
        if self.path.is_file():
            self._header = self._read_header()
            self._extras = [c for c in self._header if c not in self.columns]

    def _read_header(self) -> List[str]:
        try:
            with open(self.path, "r", newline="", encoding="utf-8") as fp:
                for row in csv.reader(fp):
                    return [str(c) for c in row]
        except OSError:
            pass
        return []

    def read_rows(self) -> List[Dict[str, str]]:
        """Read the existing rows as dicts."""
        if not self.path.is_file():
            return []
        rows: List[Dict[str, str]] = []
        try:
            with open(self.path, "r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for raw in reader:
                    if raw is None:
                        continue
                    clean = {str(k): ("" if v is None else str(v)) for k, v in raw.items() if k is not None}
                    if any(v.strip() for v in clean.values()):
                        rows.append(clean)
        except (OSError, csv.Error):
            return rows
        return rows

    def header(self) -> List[str]:
        return list(self._header) if self._header else list(self.columns)

    def reset(self) -> None:
        """Forget any header read from a previous file."""
        self._header = []
        self._extras = []

    @staticmethod
    def _render(rows: Iterable[Sequence[Any]]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])
        return buf.getvalue()

    def _ensure_header(self, header: List[str]) -> None:
        if self._header == header and self.path.is_file() and self.path.stat().st_size > 0:
            return
        rows = self.read_rows()
        text = self._render([header] + [[row.get(c, "") for c in header] for row in rows])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)
        self._header = list(header)
        self._extras = [c for c in header if c not in self.columns]

    def append(self, values: Mapping[str, Any], extras: Sequence[str] = ()) -> None:
        """Append one row."""
        header = list(self.columns) + sorted(set(self._extras) | set(extras))
        self._ensure_header(header)
        line = self._render([[values.get(c, "") for c in self._header]])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as fp:
            fp.write(line)
            fp.flush()
            if self.sync:
                try:
                    os.fsync(fp.fileno())
                except OSError:
                    pass


# Run journal for TABU search


class RunJournal:
    """
    Durable state of one tabu run
    """

    def __init__(
        self,
        csv_path: Path | str,
        *,
        objective_metrics: Sequence[str],
        state_path: Path | str | None = None,
        failure_path: Path | str | None = None,
        sync: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.objective_metrics = list(objective_metrics)
        self.state_path = (
            Path(state_path)
            if state_path
            else self.csv_path.with_suffix(".state.json")
        )
        self.failure_path = (
            Path(failure_path)
            if failure_path
            else self.csv_path.with_name(f"{self.csv_path.stem}_failed.csv")
        )
        results_columns = ["k"] + self.objective_metrics + [
            c for c in RESULTS_FIXED_COLUMNS if c != "k"
        ]
        self._results = CsvAppender(self.csv_path, results_columns, sync=sync)
        self._failures = CsvAppender(self.failure_path, FAILURE_COLUMNS, sync=sync)
        self._records: Dict[str, EvalRecord] = {}
        self._failure_rows: Dict[str, Dict[str, str]] = {}
        self._state: Optional[Dict[str, Any]] = None

    def exists(self) -> bool:
        """True when a previous run left something usable behind."""
        return any(p.exists() for p in (self.csv_path, self.state_path, self.failure_path))

    def archive(self) -> Optional[Path]:
        """Move previous artefacts aside so a fresh run cannot overwrite them."""
        if not self.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.csv_path.with_name(f"{self.csv_path.stem}.archive-{stamp}")
        counter = 1
        while target.exists():
            target = self.csv_path.with_name(f"{self.csv_path.stem}.archive-{stamp}-{counter}")
            counter += 1
        target.mkdir(parents=True, exist_ok=True)
        for path in (self.csv_path, self.state_path, self.failure_path):
            if path.exists():
                shutil.move(str(path), str(target / path.name))
        self._records.clear()
        self._failure_rows.clear()
        self._state = None
        self._results.reset()
        self._failures.reset()
        return target

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Read prior results/failures/state from disk.
        """
        self._records.clear()
        self._failure_rows.clear()
        self._state = self._load_state()

        fixed = set(self.objective_metrics) | set(RESULTS_FIXED_COLUMNS)
        for row in self._results.read_rows():
            eid = (row.get("eval_id") or "").strip()
            flags = (row.get("flags") or "").strip()
            if not eid:
                continue
            metrics: MetricDict = {}
            for key, raw in row.items():
                if key in fixed or raw in (None, ""):
                    continue
                try:
                    metrics[str(key)] = float(raw)
                except (TypeError, ValueError):
                    continue
            for name in self.objective_metrics:
                raw = row.get(name, "")
                if raw in ("", None):
                    continue
                try:
                    metrics.setdefault(name, float(raw))
                except (TypeError, ValueError):
                    continue
            value = None
            if self.objective_metrics:
                raw = row.get(self.objective_metrics[0], "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
            if value is None:
                continue
            try:
                k = int(float(row.get("k") or 0))
            except (TypeError, ValueError):
                k = 0
            self._records[eid] = EvalRecord(
                k=k,
                eval_id=eid,
                value=value,
                flags_key=row.get("compiler_flags", "") or "",
                flags=flags,
                env=self._parse_env(row.get("env", "")),
                binary=row.get("binary", "") or "",
                compile_key=row.get("compile_key", "") or "",
                metrics=metrics,
                cached=True,
            )

        for row in self._failures.read_rows():
            eid = (row.get("eval_id") or "").strip()
            if eid:
                self._failure_rows[eid] = dict(row)
        return self._state

    @staticmethod
    def _parse_env(raw: str) -> Dict[str, str]:
        try:
            parsed = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items()}

    def _load_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_path.is_file():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or int(payload.get("schema", -1)) != STATE_SCHEMA:
            return None
        return payload

    @property
    def state(self) -> Optional[Dict[str, Any]]:
        return self._state

    def lookup(self, eid: str) -> Optional[EvalRecord]:
        return self._records.get(eid)

    def failure_reason(self, eid: str) -> Optional[str]:
        row = self._failure_rows.get(eid)
        if row is None:
            return None
        return row.get("reason") or "previous failure"

    def known_records(self) -> Dict[str, EvalRecord]:
        return dict(self._records)

    def known_failures(self) -> Dict[str, Dict[str, str]]:
        return {eid: dict(row) for eid, row in self._failure_rows.items()}

    def append(self, record: EvalRecord) -> None:
        """Persist one successful evaluation immediately."""
        row: Dict[str, Any] = {
            "k": record.k,
            "eval_id": record.eval_id,
            "compiler_flags": record.flags_key,
            "flags": record.flags,
            "env": env_key(record.env),
            "binary": record.binary,
            "compile_key": record.compile_key,
            "cached": int(bool(record.cached)),
        }
        for name, value in record.metrics.items():
            row[str(name)] = value
        extras = sorted(set(record.metrics) - set(self.objective_metrics))
        self._results.append(row, extras=extras)
        self._records[record.eval_id] = record

    def record_failure(self, *, k: int, eid: str, flags: str, env: Mapping[str, str], reason: str, binary: str = "") -> None:
        """Persist a failed build/measurement immediately, then remember it."""
        short = str(reason).replace("\n", " ")[:2000]
        self._failures.append(
            {
                "k": k,
                "eval_id": eid,
                "flags": flags,
                "env": env_key(env),
                "reason": short,
                "binary": binary,
            }
        )
        self._failure_rows[eid] = {
            "k": str(k),
            "eval_id": eid,
            "flags": flags,
            "env": env_key(env),
            "reason": short,
            "binary": binary,
        }

    def save_state(
        self,
        *,
        fingerprint: str,
        search: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Atomically snapshot the search state so a restart can continue."""
        payload: Dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "study": "tabu",
            "fingerprint": fingerprint,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "search": dict(search),
        }
        if meta:
            payload["meta"] = dict(meta)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.state_path)
        self._state = payload
