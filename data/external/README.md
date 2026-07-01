# External Validation Data

Place external validation datasets here, unchanged.

Supported dataset keys:

- `aptos`
- `eyepacs`
- `messidor`
- `idrid`

Run validation with:

```powershell
python -m src.external_validation --dataset <dataset> --model <model-path> --labels-csv <labels.csv> --image-dir <images> --label-map '<json mapping>'
```

Rules:

- Keep external metrics separate from internal test metrics.
- Document label mappings in `docs/label_mapping.md`.
- Do not report external validation unless the dataset license/access terms allow it.