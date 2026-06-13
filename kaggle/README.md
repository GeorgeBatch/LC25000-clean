# kaggle/

This directory contains everything needed to share the LC25000 tile-group annotations on Kaggle.

## Contents

| File | Description |
|---|---|
| `make_image_groups_csv.py` | Builds `lc25000_image_groups.csv` from the repo's annotation files |
| `lc25000_image_groups.csv` | The consolidated group annotation CSV (upload this to Kaggle) |
| `dataset-metadata.json` | Kaggle Datasets API metadata for publishing the CSV |
| `lc25000-clean-proper-splits.ipynb` | Kaggle notebook demonstrating leakage and proper splitting |

## CSV column schema

| Column | Example | Description |
|---|---|---|
| `stem` | `lungaca1352` | Filename without extension — use this to match images |
| `filename` | `lungaca1352.jpeg` | Full filename |
| `label` | `lung_aca` | Class folder name |
| `tissue` | `lung` | Tissue type (`lung` or `colon`) |
| `local_cluster_label` | `0` | Per-class group id (0-indexed) |
| `group_id` | `743` | **Globally unique group id across all 5 classes** — use this for splitting |

There are 25,000 rows (5,000 per class) and 1,246 distinct `group_id`s.

## Rebuilding the CSV

```shell
python kaggle/make_image_groups_csv.py
```

Run from the repo root. Requires only Python stdlib — no extra packages needed.

## Publishing to Kaggle

### Step 1: install the Kaggle CLI and add your API token

```shell
pip install kaggle          # or: uv pip install kaggle
# download ~/.kaggle/kaggle.json from kaggle.com → Account → Create New API Token
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: set your Kaggle username in dataset-metadata.json

Edit the `"id"` field in `kaggle/dataset-metadata.json`:
```json
"id": "YOUR_KAGGLE_USERNAME/lc25000-clean-groups"
```

### Step 3: publish the dataset

```shell
kaggle datasets create -p kaggle/
```

### Step 4: publish the notebook

Create `kaggle/kernel-metadata.json` (fill in your username and dataset slugs), then:

```shell
kaggle kernels push -p kaggle/
```

## Quick-use snippet

```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

groups_df = pd.read_csv('/kaggle/input/lc25000-clean-groups/lc25000_image_groups.csv')

# merge group_id onto your image DataFrame by stem, then:
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['group_id']))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

assert len(set(train_df['group_id']) & set(test_df['group_id'])) == 0  # no leak
```
