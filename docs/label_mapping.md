# Label Mapping

## Project Labels

| Project label | Meaning |
|---:|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## APTOS 2019

| Raw label | Project label | Meaning |
|---:|---:|---|
| 0 | 0 | No DR |
| 1 | 1 | Mild DR |
| 2 | 2 | Moderate DR |
| 3 | 3 | Severe DR |
| 4 | 4 | Proliferative DR |

## EyePACS

EyePACS commonly uses a 0-4 diabetic retinopathy scale. Verify the exact CSV schema and pass a mapping if source labels differ.

## Messidor

Messidor label schemas vary by release and annotation file. Do not assume a 1:1 mapping. Pass `--label-map` to `src.external_validation` and document the table here before publishing metrics.

## IDRiD

IDRiD includes disease grading and lesion annotations. Use only DR severity labels for this project task unless a separate task definition is added.

Example external validation mapping:

```powershell
python -m src.external_validation --dataset messidor --model models/efficientnet_b0.keras --labels-csv data/external/messidor/labels.csv --image-dir data/external/messidor/images --label-map '{"0":0,"1":1,"2":2,"3":4}'
```