# External Validation Data

Place external validation datasets here, unchanged.

Expected layout examples:

```text
data/external/eyepacs/labels.csv
data/external/eyepacs/images/
data/external/messidor/labels.csv
data/external/messidor/images/
data/external/idrid/labels.csv
data/external/idrid/images/
```

Supported dataset keys:

- `aptos`
- `eyepacs`
- `messidor`
- `idrid`

Run validation with a documented mapping file:

```powershell
python -m src.external_validation --dataset <dataset> --model <model-path> --labels-csv <labels.csv> --image-dir <images> --label-map configs/label_mappings/<mapping>.json --site <deployment-site>
```

Outputs are written under `reports/external_validation/`:

- `<dataset>_metrics.json`
- `<dataset>_predictions.csv`
- `<site>_<model>_calibration.json`
- `<site>_<model>_calibration.png`
- `<site>_<model>_thresholds.yaml`

Rules:

- Keep external metrics separate from internal test metrics.
- Document label mappings in `docs/label_mapping.md`.
- Do not report external validation unless the dataset license/access terms allow it.
- If a source CSV does not match the documented mapping file, create a new mapping file instead of silently reusing one.