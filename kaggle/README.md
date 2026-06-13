# LC25000 Clean Image Groups

Manually-verified tile-group annotations for the [LC25000](https://arxiv.org/abs/1912.12142v1) lung and colon histopathology dataset.

LC25000 was created by augmenting 250 original tissue tiles per class (1,250 tiles total) with random rotations and flips, producing ~20 correlated copies per tile. Splitting the dataset randomly leaks copies of the same tile into both train and test sets (~98% of tile-groups are contaminated), inflating reported accuracy.

This dataset provides a single CSV (`lc25000_image_groups.csv`) assigning each of the 25,000 images to its original tile group (`group_id`), enabling proper group-aware train/test splitting.

See the companion notebook **[LC25000: Avoiding Augmentation Leakage](https://www.kaggle.com/code/gbatchkala/lc25000-avoiding-augmentation-leakage)** for a full walkthrough.

Based on: Batchkala et al., *Evaluating Histopathology Foundation Models for Few-Shot Tissue Clustering: An Application to LC25000 Augmented Dataset Cleaning*, DEMI @ MICCAI 2024 (**Best Paper Award**). DOI: [10.1007/978-3-031-73748-0_2](https://doi.org/10.1007/978-3-031-73748-0_2) | [GitHub](https://github.com/GeorgeBatch/LC25000-clean)

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

## Quick-use snippet

```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

groups_df = pd.read_csv('/kaggle/input/lc25000-clean-groups/lc25000_image_groups.csv')

# df is your image DataFrame with at least a 'stem' column (filename without extension).
# Merge group_id in, then split:
df = df.merge(groups_df[['stem', 'group_id']], on='stem', how='left')

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['group_id']))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

assert len(set(train_df['group_id']) & set(test_df['group_id'])) == 0  # no leak
```

## Citation

```bibtex
@inproceedings{batchkala2025EvaluatingHistopathologyFoundation,
  title     = {Evaluating {Histopathology Foundation Models} for~{Few-Shot Tissue Clustering}:
               {An~Application} to~{LC25000 Augmented Dataset Cleaning}},
  author    = {Batchkala, George and Li, Bin and Rittscher, Jens},
  booktitle = {Data Engineering in Medical Imaging},
  pages     = {11--21},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-73748-0_2},
}
```
