from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .constants import PROJECT_ROOT

CASE_DB = PROJECT_ROOT / "reports" / "case_history.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    run_id TEXT PRIMARY KEY,
    patient_id TEXT,
    site_id TEXT,
    generated_at TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    created_role TEXT,
    quality_status TEXT,
    prediction_status TEXT,
    prediction_class_id INTEGER,
    prediction_class_name TEXT,
    confidence REAL,
    manual_review INTEGER,
    report_path TEXT,
    gradcam_path TEXT,
    model_path TEXT,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_generated_at ON cases(generated_at);
CREATE INDEX IF NOT EXISTS idx_cases_patient_id ON cases(patient_id);
CREATE INDEX IF NOT EXISTS idx_cases_site_id ON cases(site_id);
"""


def _connect(db_path: str | Path = CASE_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_case(result: Mapping[str, Any], principal: Mapping[str, str] | None = None, db_path: str | Path = CASE_DB) -> str:
    metadata = result.get("metadata", {})
    prediction = result.get("prediction", {})
    quality = result.get("quality", {})
    uncertainty = result.get("uncertainty", {})
    outputs = result.get("outputs", {})
    model = result.get("model", {})
    run_id = metadata.get("run_id")
    if not run_id:
        raise ValueError("screening result is missing metadata.run_id")

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cases (
                run_id, patient_id, site_id, generated_at, created_at, created_by, created_role,
                quality_status, prediction_status, prediction_class_id, prediction_class_name,
                confidence, manual_review, report_path, gradcam_path, model_path, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                metadata.get("patient_id"),
                metadata.get("site_id"),
                metadata.get("generated_at"),
                datetime.now().isoformat(timespec="seconds"),
                (principal or {}).get("id"),
                (principal or {}).get("role"),
                quality.get("status"),
                prediction.get("status"),
                prediction.get("class_id"),
                prediction.get("class_name"),
                prediction.get("confidence"),
                int(bool(uncertainty.get("manual_review"))),
                outputs.get("report_path"),
                outputs.get("gradcam_path") or outputs.get("explanation_path"),
                model.get("active_path"),
                json.dumps(result, allow_nan=False),
            ),
        )
    return str(run_id)


def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "patient_id": row["patient_id"],
        "site_id": row["site_id"],
        "generated_at": row["generated_at"],
        "created_at": row["created_at"],
        "quality_status": row["quality_status"],
        "prediction_status": row["prediction_status"],
        "prediction_class_id": row["prediction_class_id"],
        "prediction_class_name": row["prediction_class_name"],
        "confidence": row["confidence"],
        "manual_review": bool(row["manual_review"]),
        "report_path": row["report_path"],
        "gradcam_path": row["gradcam_path"],
        "model_path": row["model_path"],
    }


def list_cases(
    limit: int = 50,
    patient_id: str | None = None,
    site_id: str | None = None,
    db_path: str | Path = CASE_DB,
) -> list[dict[str, Any]]:
    clauses = []
    values: list[Any] = []
    if patient_id:
        clauses.append("patient_id = ?")
        values.append(patient_id)
    if site_id:
        clauses.append("site_id = ?")
        values.append(site_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values.append(max(1, min(int(limit), 500)))
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM cases {where} ORDER BY COALESCE(generated_at, created_at) DESC LIMIT ?",
            values,
        ).fetchall()
    return [_row_to_summary(row) for row in rows]


def get_case(run_id: str, db_path: str | Path = CASE_DB) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT result_json FROM cases WHERE run_id = ?", (run_id,)).fetchone()
    return None if row is None else json.loads(row["result_json"])