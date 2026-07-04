# Label Mapping

## Project Labels

| Project label | Meaning |
|---:|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## Mapping Files

Reusable mappings live under `configs/label_mappings/`:

| Dataset/source schema | Mapping file | Notes |
|---|---|---|
| APTOS 0-4 DR grade | `configs/label_mappings/aptos_5class.json` | Direct project scale. |
| EyePACS/Kaggle 0-4 DR grade | `configs/label_mappings/eyepacs_5class.json` | Direct project scale when the CSV `level` field uses 0-4 severity. |
| IDRiD disease grading 0-4 | `configs/label_mappings/idrid_5class.json` | Direct project scale for DR grading labels. |
| Messidor retinopathy grade 0-3 | `configs/label_mappings/messidor_retinopathy_grade_0_3.json` | Maps 3 to project label 4 because this schema does not split severe NPDR from proliferative DR. Override if your CSV has a richer schema. |

The loader registry in `src.datasets.DATASET_SPECS` is the code source of truth. `docs` and validation commands should match it.

## Validation Examples

```powershell
python -m src.external_validation --dataset eyepacs --model models/efficientnet_b0.keras --labels-csv data/external/eyepacs/labels.csv --image-dir data/external/eyepacs/images --label-map configs/label_mappings/eyepacs_5class.json --site eyepacs_site
python -m src.external_validation --dataset messidor --model models/efficientnet_b0.keras --labels-csv data/external/messidor/labels.csv --image-dir data/external/messidor/images --label-map configs/label_mappings/messidor_retinopathy_grade_0_3.json --site messidor_site
python -m src.external_validation --dataset idrid --model models/efficientnet_b0.keras --labels-csv data/external/idrid/labels.csv --image-dir data/external/idrid/images --label-map configs/label_mappings/idrid_5class.json --site idrid_site
```

External validation writes metrics, prediction rows, calibration plots, and tuned site thresholds under `reports/external_validation/`. Do not publish external metrics until the dataset files and source label schema have been verified.