from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "fastapi_backend" / "state"
DB_PATH = STATE_DIR / "jobs.db"
TERMINAL_STATUSES = {"completed", "failed", "stopped"}
OUTPUTS_ROOT = REPO_ROOT / "outputs"


def _utcnow() -> datetime:
    return datetime.utcnow()


def _serialize_dt(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _deserialize_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def _pid_is_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    script_name: str
    command: list[str]
    params: dict[str, str]
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    result_csv: Optional[str] = None
    status: str = "queued"
    pid: Optional[int] = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)
    log_handle: Optional[object] = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "script_name": self.script_name,
            "command": list(self.command),
            "params": dict(self.params),
            "output_dir": self.output_dir,
            "log_file": self.log_file,
            "result_csv": self.result_csv,
            "status": self.status,
            "pid": self.pid,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "error_message": self.error_message,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "JobRecord":
        return cls(
            job_id=row["job_id"],
            job_type=row["job_type"],
            script_name=row["script_name"],
            command=json.loads(row["command_json"]),
            params=json.loads(row["params_json"]),
            output_dir=row["output_dir"],
            log_file=row["log_file"],
            result_csv=row["result_csv"],
            status=row["status"],
            pid=row["pid"],
            created_at=_deserialize_dt(row["created_at"]) or _utcnow(),
            started_at=_deserialize_dt(row["started_at"]),
            finished_at=_deserialize_dt(row["finished_at"]),
            return_code=row["return_code"],
            error_message=row["error_message"],
        )


class JobManager:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._db_path = db_path
        self._init_db()
        self._load_jobs()
        self._import_existing_outputs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    script_name TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    output_dir TEXT,
                    log_file TEXT,
                    result_csv TEXT,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    error_message TEXT
                )
                """
            )
            connection.commit()

    def _load_jobs(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs").fetchall()

        with self._lock:
            for row in rows:
                record = JobRecord.from_row(row)
                self._jobs[record.job_id] = record

            jobs = list(self._jobs.values())

        for job in jobs:
            self._recover_job(job)

    def _recover_job(self, record: JobRecord) -> None:
        if record.status in TERMINAL_STATUSES:
            return

        if _pid_is_alive(record.pid):
            record.status = "running"
            self._persist_record(record)
            return

        record.finished_at = record.finished_at or _utcnow()
        if record.result_csv and Path(record.result_csv).exists():
            record.status = "completed"
        else:
            record.status = "failed"
            if not record.error_message:
                record.error_message = (
                    "Recovered after FastAPI restart; original process was not found."
                )
        self._persist_record(record)

    def _persist_record(self, record: JobRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, script_name, command_json, params_json,
                    output_dir, log_file, result_csv, status, pid,
                    created_at, started_at, finished_at, return_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type=excluded.job_type,
                    script_name=excluded.script_name,
                    command_json=excluded.command_json,
                    params_json=excluded.params_json,
                    output_dir=excluded.output_dir,
                    log_file=excluded.log_file,
                    result_csv=excluded.result_csv,
                    status=excluded.status,
                    pid=excluded.pid,
                    created_at=excluded.created_at,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    return_code=excluded.return_code,
                    error_message=excluded.error_message
                """,
                (
                    record.job_id,
                    record.job_type,
                    record.script_name,
                    json.dumps(record.command),
                    json.dumps(record.params),
                    record.output_dir,
                    record.log_file,
                    record.result_csv,
                    record.status,
                    record.pid,
                    _serialize_dt(record.created_at),
                    _serialize_dt(record.started_at),
                    _serialize_dt(record.finished_at),
                    record.return_code,
                    record.error_message,
                ),
            )
            connection.commit()

    def _find_existing_record(
        self,
        *,
        log_file: Optional[str],
        output_dir: Optional[str],
        script_name: str,
    ) -> Optional[JobRecord]:
        with self._lock:
            for record in self._jobs.values():
                if log_file and record.log_file == log_file:
                    return record
                if output_dir and record.output_dir == output_dir and record.script_name == script_name:
                    return record
        return None

    def _infer_job_type(self, category: str) -> tuple[str, str]:
        if category.endswith("_lora_ablation"):
            return "ablation", category[: -len("_lora_ablation")]
        if category.endswith("_lora_imbalance"):
            return "llm_imbalance", category[: -len("_lora_imbalance")]
        if category.endswith("_lora"):
            return "llm", category[: -len("_lora")]
        if category.endswith("_imbalance"):
            return "classical_imbalance", category[: -len("_imbalance")]
        return "classical", category

    def _infer_action_and_script(self, job_type: str, log_name: str) -> tuple[Optional[str], Optional[str]]:
        if job_type in {"classical", "classical_imbalance"}:
            if log_name.startswith("train_"):
                return "train", "train.sh" if job_type == "classical" else "train_imbalance.sh"
            if log_name.startswith("test_"):
                return "test", "test.sh" if job_type == "classical" else "test_imbalance.sh"
            return None, None

        if job_type == "llm":
            if log_name.startswith("finetuning_"):
                return "finetune", "finetune.sh"
            if log_name.startswith("inference_"):
                return "inference", "inference.sh"
            return None, None

        if job_type == "llm_imbalance":
            if log_name.startswith("finetuning_"):
                return "finetune", "finetune_imbalance.sh"
            if log_name.startswith("inference_"):
                return "inference", "inference_imbalance.sh"
            return None, None

        if job_type == "ablation":
            if log_name.startswith("finetuning_"):
                return "finetune", "finetune_ablation.sh"
            if log_name.startswith("inference_"):
                return "inference", "inference_ablation.sh"
            return None, None

        return None, None

    def _infer_params(
        self,
        *,
        job_type: str,
        action: str,
        model_name: str,
        dirname: str,
    ) -> dict[str, str]:
        params: dict[str, str] = {"action": action, "model_name": model_name}

        if job_type == "classical":
            dataset_name, length = dirname.rsplit("_", 1)
            params["dataset_name"] = dataset_name
            params["length"] = length
            return params

        if job_type == "classical_imbalance":
            dataset_name, length, pos_ratio = dirname.rsplit("_", 2)
            params["dataset_name"] = dataset_name
            params["length"] = length
            params["pos_ratio"] = pos_ratio
            return params

        if job_type == "llm":
            dataset_name, length = dirname.rsplit("_", 1)
            params["dataset_name"] = dataset_name
            params["length"] = length
            return params

        if job_type == "llm_imbalance":
            dataset_name, pos_ratio = dirname.rsplit("_", 1)
            params["dataset_name"] = dataset_name
            params["pos_ratio"] = pos_ratio
            return params

        if job_type == "ablation":
            dataset_name, r_value, alpha_value = dirname.rsplit("_", 2)
            params["dataset_name"] = dataset_name
            params["r"] = r_value
            params["alpha"] = alpha_value
            return params

        return params

    def _import_record_from_log(self, log_path: Path) -> None:
        output_dir = log_path.parent
        category = output_dir.parent.name
        dirname = output_dir.name
        job_type, model_name = self._infer_job_type(category)
        action, script_name = self._infer_action_and_script(job_type, log_path.name)
        if action is None or script_name is None:
            return

        log_file = str(log_path.resolve())
        output_dir_str = str(output_dir.resolve())
        if self._find_existing_record(log_file=log_file, output_dir=output_dir_str, script_name=script_name):
            return

        try:
            params = self._infer_params(
                job_type=job_type,
                action=action,
                model_name=model_name,
                dirname=dirname,
            )
        except ValueError:
            return

        script_path = REPO_ROOT / "scripts" / script_name
        command = ["bash", str(script_path)]
        result_csv_path = output_dir / "results.csv"
        result_csv = str(result_csv_path.resolve()) if result_csv_path.exists() else None
        timestamp = _utcnow()
        if log_path.exists():
            timestamp = datetime.utcfromtimestamp(log_path.stat().st_mtime)
        elif result_csv_path.exists():
            timestamp = datetime.utcfromtimestamp(result_csv_path.stat().st_mtime)

        record = JobRecord(
            job_id=uuid.uuid4().hex,
            job_type=job_type,
            script_name=script_name,
            command=command,
            params=params,
            output_dir=output_dir_str,
            log_file=log_file,
            result_csv=result_csv,
            status="completed",
            created_at=timestamp,
            started_at=timestamp,
            finished_at=timestamp,
        )

        with self._lock:
            self._jobs[record.job_id] = record
        self._persist_record(record)

    def _import_existing_outputs(self) -> None:
        if not OUTPUTS_ROOT.exists():
            return

        for log_path in sorted(OUTPUTS_ROOT.glob("*/*/*.log")):
            self._import_record_from_log(log_path)

    def start_job(
        self,
        *,
        job_type: str,
        script_name: str,
        command: list[str],
        params: dict[str, str],
        output_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
        result_csv: Optional[Path] = None,
        working_dir: Optional[Path] = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            script_name=script_name,
            command=command,
            params=params,
            output_dir=str(output_dir) if output_dir else None,
            log_file=str(log_file) if log_file else None,
            result_csv=str(result_csv) if result_csv else None,
        )

        with self._lock:
            self._jobs[job_id] = record
        self._persist_record(record)

        log_handle = None
        try:
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_file, "ab")
            process = subprocess.Popen(
                command,
                cwd=working_dir or REPO_ROOT,
                stdout=log_handle if log_handle is not None else subprocess.DEVNULL,
                stderr=log_handle if log_handle is not None else subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            if log_handle is not None:
                log_handle.close()
            record.status = "failed"
            record.error_message = str(exc)
            record.finished_at = _utcnow()
            self._persist_record(record)
            return record

        record.process = process
        record.log_handle = log_handle
        record.pid = process.pid
        record.status = "running"
        record.started_at = _utcnow()
        self._persist_record(record)

        watcher = threading.Thread(target=self._watch_job, args=(job_id,), daemon=True)
        watcher.start()
        return record

    def _watch_job(self, job_id: str) -> None:
        record = self.get_job(job_id)
        if record is None or record.process is None:
            return

        return_code = record.process.wait()
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            current.return_code = return_code
            current.finished_at = _utcnow()
            current.pid = None
            if current.status != "stopped":
                current.status = "completed" if return_code == 0 else "failed"
            if current.log_handle is not None:
                current.log_handle.close()
                current.log_handle = None
            self._persist_record(current)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())

        for job in jobs:
            self._sync_status(job)
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            self._sync_status(job)
        return job

    def stop_job(self, job_id: str) -> Optional[JobRecord]:
        record = self.get_job(job_id)
        if record is None or record.pid is None:
            return record

        if _pid_is_alive(record.pid):
            try:
                os.killpg(os.getpgid(record.pid), 15)
            except ProcessLookupError:
                pass

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return record
            current.status = "stopped"
            current.finished_at = _utcnow()
            current.return_code = current.process.poll() if current.process else current.return_code
            current.pid = None
            if current.log_handle is not None:
                current.log_handle.close()
                current.log_handle = None
            self._persist_record(current)
            return current

    def _sync_status(self, record: JobRecord) -> None:
        process = record.process
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                return

            with self._lock:
                current = self._jobs.get(record.job_id)
                if current is None or current.status in TERMINAL_STATUSES:
                    return
                current.return_code = return_code
                current.finished_at = current.finished_at or _utcnow()
                current.status = "completed" if return_code == 0 else "failed"
                current.pid = None
                if current.log_handle is not None:
                    current.log_handle.close()
                    current.log_handle = None
                self._persist_record(current)
            return

        if record.status in TERMINAL_STATUSES:
            return

        if _pid_is_alive(record.pid):
            return

        with self._lock:
            current = self._jobs.get(record.job_id)
            if current is None or current.status in TERMINAL_STATUSES:
                return
            current.finished_at = current.finished_at or _utcnow()
            if current.result_csv and Path(current.result_csv).exists():
                current.status = "completed"
            else:
                current.status = "failed"
                current.error_message = current.error_message or (
                    "Process ended without a tracked return code."
                )
            current.pid = None
            self._persist_record(current)
