import json
from pathlib import Path

import numpy as np

from src.calibration import confidence_calibration, tune_uncertainty_thresholds, write_calibration_artifacts
from src.datasets import dataset_mapping_documentation


def test_calibration_curve_and_threshold_tuning(work_dir):
    y_true = np.asarray([0, 1, 2, 0, 3, 4], dtype=int)
    probabilities = np.asarray(
        [
            [0.92, 0.02, 0.02, 0.02, 0.02],
            [0.60, 0.25, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.78, 0.08, 0.04],
            [0.70, 0.10, 0.10, 0.05, 0.05],
            [0.05, 0.10, 0.10, 0.70, 0.05],
            [0.05, 0.05, 0.05, 0.10, 0.75],
        ],
        dtype=float,
    )
    curve = confidence_calibration(y_true, probabilities, n_bins=5)
    assert curve["n"] == 6
    assert 0 <= curve["ece"] <= 1

    tuned = tune_uncertainty_thresholds(y_true, probabilities, site="clinic-a", max_false_negative_rate=0.25)
    assert tuned["site"] == "clinic-a"
    assert set(tuned["thresholds"]) == {"min_confidence", "max_entropy", "min_top2_margin"}

    artifacts = write_calibration_artifacts(y_true, probabilities, site="clinic-a", model_name="demo", out_dir=work_dir)
    for path in artifacts.values():
        assert Path(path).exists()
    payload = json.loads(Path(artifacts["calibration_json"]).read_text(encoding="utf-8"))
    assert payload["threshold_tuning"]["site"] == "clinic-a"


def test_dataset_mapping_documentation_contains_external_sets():
    docs = dataset_mapping_documentation()
    assert docs["eyepacs"]["label_mapping"]["4"] == 4
    assert docs["idrid"]["label_mapping"]["3"] == 3
    assert docs["messidor"]["label_mapping"]["3"] == 4