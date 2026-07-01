# Raw Data

Keep downloaded datasets here, unchanged. Do not commit retinal images or competition data.

Expected first dataset layout for APTOS 2019:

```text
data/raw/aptos2019/
|-- train.csv
`-- images_288_scaled/
```

The training CLI expects `train.csv` with `id_code` and `diagnosis` columns.

