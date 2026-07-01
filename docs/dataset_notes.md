# Dataset Notes

Supported dataset keys:

- `aptos` / `aptos2019`
- `eyepacs`
- `messidor`
- `idrid`

Raw data stays under `data/raw/` or `data/external/` and should not be committed.

All loaders normalize to this dataframe contract:

```text
dataset | image_id | path | label
```

Project labels:

- 0: No DR
- 1: Mild DR
- 2: Moderate DR
- 3: Severe DR
- 4: Proliferative DR

External datasets must document source labels, mapping, access rules, and license constraints before metrics are reported. Internal test metrics and external validation metrics must stay separate.