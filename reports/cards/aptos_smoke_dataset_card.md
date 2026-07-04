# RetinaAI Dataset Card: aptos_smoke

## Source Contract

- Labels CSV: `data/raw/aptos2019/train.csv`
- Image directory: `data/raw/aptos2019/images_288_scaled`
- Loader key: `aptos_smoke`
- Mapping notes: Custom dataset mapping supplied by caller.

## Label Mapping

```json
{}
```

## Split Distributions

```json
{
  "test": {
    "0": 2,
    "1": 1,
    "3": 1,
    "4": 1
  },
  "train": {
    "0": 10,
    "1": 1,
    "2": 4,
    "3": 3,
    "4": 3
  },
  "validation": {
    "0": 2,
    "2": 1,
    "4": 1
  }
}
```

## Use Constraints

Raw dataset files stay outside git. Before publishing external validation metrics, confirm dataset license/access terms and document any source-schema conversion used for the labels.
