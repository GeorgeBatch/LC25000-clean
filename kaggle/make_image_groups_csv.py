"""
Build a single consolidated CSV that assigns every one of LC25000's 25,000 images
to its original tile group (group_id).

The LC25000 dataset was created by augmenting 250 original tissue tiles per class
(1,250 tiles total) with random rotations and flips, producing 20 augmented copies
per tile.  Splitting the dataset randomly ignores this structure and leaks copies of
the same tile into both train and test sets (see MICCAI-2024 paper).

This script reads the manually-verified per-class annotation files produced by the
semi-automatic clustering pipeline (annotations/<class>/UNI/resize_only/final_clusters.csv)
and writes kaggle/lc25000_image_groups.csv with the following columns:

    stem               – filename without extension, e.g. "lungaca1352"
    filename           – full filename,              e.g. "lungaca1352.jpeg"
    label              – class folder name,          e.g. "lung_aca"
    tissue             – tissue type,                e.g. "lung"
    local_cluster_label – per-class group id (0-indexed, 0..n_clusters-1)
    group_id           – globally unique group id across all 5 classes

Usage:
    python kaggle/make_image_groups_csv.py
"""

import csv
import os

# ---------------------------------------------------------------------------
# Constants (inlined from source/constants.py so this script is self-contained)
# ---------------------------------------------------------------------------
ALL_CANCER_TYPES = ('colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc')
NUM_TOTAL_IMAGES = 25_000
NUM_CLASS_IMAGES = 5_000

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_SAVE_DIR = os.path.join(REPO_ROOT, 'annotations')
OUTPUT_CSV = os.path.join(REPO_ROOT, 'kaggle', 'lc25000_image_groups.csv')

# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------

rows = []
all_stems = []
current_max_plus_one = 0  # running offset to make group_id globally unique

for cancer_type in ALL_CANCER_TYPES:
    csv_path = os.path.join(
        ANNOTATIONS_SAVE_DIR, cancer_type, 'UNI', 'resize_only', 'final_clusters.csv'
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Annotation file not found: {csv_path}\n"
            "Please run from the repo root after ensuring the annotations/ directory is present."
        )

    class_rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_cluster_label = int(row['cluster_label'])
            img_path = row['img_path']
            filename = os.path.basename(img_path)
            stem = os.path.splitext(filename)[0]
            tissue = cancer_type.split('_')[0]

            class_rows.append({
                'stem': stem,
                'filename': filename,
                'label': cancer_type,
                'tissue': tissue,
                'local_cluster_label': local_cluster_label,
                'group_id': current_max_plus_one + local_cluster_label,
            })

    if len(class_rows) != NUM_CLASS_IMAGES:
        raise ValueError(
            f"{cancer_type}: expected {NUM_CLASS_IMAGES} rows, got {len(class_rows)}"
        )

    # Verify this class's global group_id block doesn't overlap with previous classes
    local_max = max(r['local_cluster_label'] for r in class_rows)
    global_ids_this_class = {r['group_id'] for r in class_rows}
    global_ids_previous = {r['group_id'] for r in rows}
    overlap = global_ids_this_class & global_ids_previous
    if overlap:
        raise AssertionError(
            f"{cancer_type}: group_id overlap with previous classes — {sorted(overlap)[:5]}"
        )

    rows.extend(class_rows)
    current_max_plus_one += local_max + 1

    stems_this_class = [r['stem'] for r in class_rows]
    all_stems.extend(stems_this_class)
    print(f"  {cancer_type}: {len(class_rows)} images, {local_max + 1} groups")

# ---------------------------------------------------------------------------
# Global sanity checks
# ---------------------------------------------------------------------------

if len(rows) != NUM_TOTAL_IMAGES:
    raise ValueError(f"Expected {NUM_TOTAL_IMAGES} rows, got {len(rows)}")

if len(set(all_stems)) != NUM_TOTAL_IMAGES:
    raise ValueError(
        f"Stems are not unique: {NUM_TOTAL_IMAGES} images but only "
        f"{len(set(all_stems))} distinct stems"
    )

num_groups = len({r['group_id'] for r in rows})
print(f"\nTotal: {len(rows)} images in {num_groups} groups across {len(ALL_CANCER_TYPES)} classes")

# ---------------------------------------------------------------------------
# Write output sorted by group_id then stem (deterministic)
# ---------------------------------------------------------------------------

rows_sorted = sorted(rows, key=lambda r: (r['group_id'], r['stem']))
fieldnames = ['stem', 'filename', 'label', 'tissue', 'local_cluster_label', 'group_id']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_sorted)

print(f"Written: {OUTPUT_CSV}")
