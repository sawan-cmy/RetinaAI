# Training Data Download

Primary dataset: APTOS 2019 Blindness Detection.

The official download requires a Kaggle account, accepted competition rules, and Kaggle CLI authentication.

## Authenticate

OAuth path:

```powershell
python -m kaggle auth login
```

API token path:

1. Open Kaggle account settings.
2. Generate an API token.
3. Save the token where Kaggle CLI expects it, then rerun the download script.

## Download

```powershell
.\scripts\download_training_data.ps1
```

Current local training source: images from Kaggle dataset `uiiurz1/aptos2019-dataset`, labels from Kaggle dataset `paraspatil/aptos2019blindnessdetection`. The official competition API still returned 403 for direct downloads.`r`n`r`nExpected output:

```text
data/raw/aptos2019/
|-- train.csv
|-- sample_submission.csv
|-- images_288_scaled/
`-- test_images/
```

Do not commit downloaded images or Kaggle archives.

