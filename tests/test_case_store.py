import pytest
from fastapi import HTTPException

from src.api import principal, require_role
from src.case_store import get_case, list_cases, save_case


def _case(run_id="run-1"):
    return {
        "metadata": {"run_id": run_id, "patient_id": "patient-1", "site_id": "clinic-a", "generated_at": "2026-07-02T10:00:00"},
        "quality": {"status": "accepted"},
        "prediction": {"status": "available", "class_id": 2, "class_name": "Moderate DR", "confidence": 0.82},
        "uncertainty": {"manual_review": False},
        "model": {"active_path": "models/demo.keras"},
        "outputs": {"report_path": "reports/demo.pdf", "gradcam_path": "reports/demo.png"},
    }


def test_case_store_round_trip(work_dir):
    db = work_dir / "cases.sqlite3"
    save_case(_case(), principal={"id": "tester", "role": "clinician"}, db_path=db)
    rows = list_cases(site_id="clinic-a", db_path=db)
    assert len(rows) == 1
    assert rows[0]["prediction_class_name"] == "Moderate DR"
    assert get_case("run-1", db_path=db)["metadata"]["patient_id"] == "patient-1"


def test_api_key_roles(monkeypatch):
    monkeypatch.setenv("RETINAAI_API_KEYS", '{"viewer-secret":"viewer","clinician-secret":"clinician"}')
    assert principal("viewer-secret")["role"] == "viewer"
    with pytest.raises(HTTPException):
        principal(None)

    clinician_gate = require_role("clinician")
    assert clinician_gate({"id": "c", "role": "clinician"})["role"] == "clinician"
    with pytest.raises(HTTPException):
        clinician_gate({"id": "v", "role": "viewer"})