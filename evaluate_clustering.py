# built-in imports
import argparse
import os
import json
from copy import deepcopy

# standard imports
import numpy as np
import pandas as pd

# local imports
from source.constants import RANDOM_SEED
from source.constants import FEATURE_VECTORS_SAVE_DIR, ANNOTATIONS_SAVE_DIR
from source.constants import ALL_CANCER_TYPES, ALL_IMG_NORMS, ALL_EXTRACTOR_MODELS, ALL_DIMENSIONALITY_REDUCTION_METHODS, ALL_CLUSTERING_ALGORITHMS, ALL_DISTANCE_METRICS
from source.eval_utils import reduce_feature_dimensionality, get_clustering_labels, compute_connectivity_matrix, compute_clustering_metrics, precision_at_1, precision_at_k


def get_imgpaths_2_intids(ids_2_imgpaths):
    assert len(set(ids_2_imgpaths.values())) == len(list(ids_2_imgpaths.values())), "Can only reverse a bijective mapping, duplicate values found."
    imgpaths_2_intids = {v: int(k) for k, v in ids_2_imgpaths.items()}
    return imgpaths_2_intids


def extract_connectivity_vector(connectivity_matrix):
    """
    Extract a 1D connectivity vector of shape (1, n*(n-1)/2) from a 2D connectivity matrix (n*n).
    It only makes sence to consider the upper triangular part of the matrix excluding the diagonal (k=1)
    Get these elements and put into 1d array.

    Parameters:
    connectivity_matrix (array-like): 2D connectivity matrix of shape (n, n)

    Returns:
    array-like: 1D connectivity vector of shape (1, n*(n-1)/2)
    """
    connectivity_vector = connectivity_matrix[np.triu_indices(
        connectivity_matrix.shape[0], k=1)]
    return connectivity_vector


def get_true_connectivity(
    manual_annotations_dir,
    ids_2_imgpaths,
):
    imgpaths_2_intids = get_imgpaths_2_intids(ids_2_imgpaths=ids_2_imgpaths)

    # true clustering was saved as json in a dictionary format: {'0': ['pathA', 'pathB'], '1': ['pathC', 'pathD']}
    with open(f"{manual_annotations_dir}/final_clusters.json", "r") as f:
        annotated_clusters_dict_str = json.load(f)
    # replace string keys with int keys
    annotated_clusters_dict = {
        int(k): deepcopy(v) for k, v in annotated_clusters_dict_str.items()}
    num_true_clusters = len(annotated_clusters_dict)
    num_total_images = sum([len(cluster)
                            for cluster in annotated_clusters_dict.values()])

    # compute cluster labels vector and connectivity matrix, images ordered in the same way as the features array
    true_cluster_labels = -1 * np.ones(num_total_images, dtype=int)
    for cluster_id, img_paths in annotated_clusters_dict.items():
        for img_path in img_paths:
            true_cluster_labels[imgpaths_2_intids[img_path]
                                ] = cluster_id
    assert np.all(true_cluster_labels != -1)
    # connectivity_matrix[i, j] is 1 if samples (i) and (j) are in the same cluster, 0 - if in different clusters
    true_connectivity_matrix = compute_connectivity_matrix(
        clusters_dict=annotated_clusters_dict, imgpaths_2_intids=imgpaths_2_intids)
    true_connectivity_vector = extract_connectivity_vector(true_connectivity_matrix)

    return true_connectivity_matrix, true_connectivity_vector, true_cluster_labels, num_true_clusters


def get_predicted_connectivity(
    predicted_cluster_labels,
    ids_2_imgpaths,
):
    imgpaths_2_intids = get_imgpaths_2_intids(ids_2_imgpaths=ids_2_imgpaths)

    # compute clusters dict - same format as annotated_clusters_dict
    predicted_clusters_dict = {}
    for i, predicted_cluster_label in enumerate(predicted_cluster_labels):
        if predicted_cluster_label not in predicted_clusters_dict:
            predicted_clusters_dict[predicted_cluster_label] = []
        # append in any case
        predicted_clusters_dict[predicted_cluster_label].append(
            ids_2_imgpaths[str(i)])
    # connectivity_matrix[i, j] is 1 if samples (i) and (j) are in the same cluster, 0 - if in different clusters
    predicted_connectivity_matrix = compute_connectivity_matrix(
        clusters_dict=predicted_clusters_dict, imgpaths_2_intids=imgpaths_2_intids)

    predicted_connectivity_vector = extract_connectivity_vector(
        predicted_connectivity_matrix)

    return predicted_connectivity_vector


# TODO: split into separate clustering and evaluation - to be used here, in 3-evaluation.ipynb, 2-clustering-interactive.ipynb
def evaluate_clustering(
    features_save_dir,
    manual_annotations_dir,
    dimensionality_reduction,
    clustering,
    distance_metric,
    verbose,
):
    print("Features path:", features_save_dir)
    print("Dimensionality reduction:", dimensionality_reduction)
    print("Clustering:", clustering)
    print("Distance metric:", distance_metric)
    print()

    features_npy_path = f'{features_save_dir}/features.npy'
    ids_2_imgpaths_json_path = f'{features_save_dir}/ids_2_img_paths.json'

    assert os.path.isfile(
        features_npy_path), f"File does not exist: \n\t{features_npy_path}"
    assert os.path.isfile(
        ids_2_imgpaths_json_path), f"File does not exist: \n\t{ids_2_imgpaths_json_path}"
    # load
    features = np.load(features_npy_path)
    if distance_metric == 'cosine':
        # after normalisation, euclidean distance is equivalent to cosine distance
        # KMeans does not support cosine distance, so we can't just pass distance_metric to KMeans as a parameter
        features = features / \
            np.linalg.norm(features, axis=1, keepdims=True)
    with open(ids_2_imgpaths_json_path, 'r') as f:
        ids_2_imgpaths = json.load(f)
    # check that the values are unique, this will allow bijective mapping
    assert len(set(ids_2_imgpaths.values())) == len(
        ids_2_imgpaths.values())

    # true connectivity info
    true_connectivity_matrix, true_connectivity_vector, true_cluster_labels, num_true_clusters = get_true_connectivity(
        manual_annotations_dir=manual_annotations_dir,
        ids_2_imgpaths=ids_2_imgpaths,
    )

    # dimensionality reduction
    features_reduced = reduce_feature_dimensionality(features=features, method=dimensionality_reduction)

    # compute precision@1 and precision@5; use euclidean distance - already normalised before
    precision_at_1_value = precision_at_1(features_reduced, true_connectivity_matrix, metric='euclidean')
    precision_at_5_value = precision_at_k(features_reduced, true_connectivity_matrix, k=5, metric='euclidean')

    # clustering
    predicted_cluster_labels = get_clustering_labels(
        features=features_reduced,
        n_clusters=num_true_clusters,
        method=clustering,
        random_state=RANDOM_SEED,
    )

    # predicted connectivity info
    predicted_connectivity_vector = get_predicted_connectivity(
        predicted_cluster_labels=predicted_cluster_labels,
        ids_2_imgpaths=ids_2_imgpaths,
    )
    assert predicted_connectivity_vector.shape == true_connectivity_vector.shape

    # Compute metrics
    metrics = compute_clustering_metrics(
        true_connectivity_vector, predicted_connectivity_vector, true_cluster_labels, predicted_cluster_labels)
    metrics['precision@1'] = precision_at_1_value
    metrics['precision@5'] = precision_at_5_value

    if verbose:
        for metric, value in metrics.items():
            if isinstance(value, int):
                print(f"{metric}: {value}")
            elif isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}:\n {value}")
        print()

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate extractor-reduction-clustering pipeline.')
    parser.add_argument('--cancer_type', type=str,
                        default='lung_aca',
                        choices=ALL_CANCER_TYPES,
                        help='Cancer type name name.')
    parser.add_argument('--extractor_name', type=str,
                        default='all',
                        choices=list(ALL_EXTRACTOR_MODELS) + ['all'],
                        help='Feature extractor name.')
    parser.add_argument('--img_norm', type=str,
                        default='all',
                        choices=list(ALL_IMG_NORMS) + ['all'])
    parser.add_argument('--distance_metric', type=str,
                        default='all', choices=list(ALL_DISTANCE_METRICS) + ['all'],
                        help='Distance metric to use for clustering evaluation.')
    parser.add_argument('--dimensionality_reduction', type=str,
                        default='all',
                        choices=list(ALL_DIMENSIONALITY_REDUCTION_METHODS) + ['all'],
                        help='Dimensionality reduction algorithm name.')
    parser.add_argument('--clustering', type=str,
                        default='all',
                        choices=list(ALL_CLUSTERING_ALGORITHMS) + ['all'],
                        help='Clustering algorithm name.')
    parser.add_argument('--manual_annotations_dir', type=str,
                        default=None,
                        help='Directory path with manual annotations.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print metrics to stdout.')
    args = parser.parse_args()

    if args.manual_annotations_dir is not None:
        assert os.path.isdir(args.manual_annotations_dir)
        assert f'/{args.cancer_type}/' in args.manual_annotations_dir, f"--cancer_type={args.cancer_type} must match the --manual_annotations_dir={args.manual_annotations_dir}"
        manual_annotations_dir = args.manual_annotations_dir
    else:
        manual_annotations_dir = f'{ANNOTATIONS_SAVE_DIR}/{args.cancer_type}/UNI/resize_only'
    assert os.path.isdir(manual_annotations_dir)

    # we can do all combinations of distance metrics, dimensionality reductions and clustering algorithms
    distance_metrics = [args.distance_metric] if args.distance_metric != 'all' else ALL_DISTANCE_METRICS
    dimensionality_reductions = [args.dimensionality_reduction] if args.dimensionality_reduction != 'all' else ALL_DIMENSIONALITY_REDUCTION_METHODS
    clusterings = [args.clustering] if args.clustering != 'all' else ALL_CLUSTERING_ALGORITHMS

    all_features_save_dir = f"{FEATURE_VECTORS_SAVE_DIR}/{args.cancer_type}"
    # use directories with precomputed features instead of ALL_EXTRACTOR_MODELS, we will not compute features here
    extractor_names = [args.extractor_name] if args.extractor_name != 'all' else os.listdir(all_features_save_dir)
    # img_norms is defined later, after we know the extractor_name

    eval_results_dir = 'eval_results'
    # should overwrite the same one file for a specific cancer type
    filename_base = f"cancer_type={args.cancer_type}#extractor_name=all#img_norm=all#distance_metric=all#dimensionality_reduction=all#clustering=all"

    # Load existing data - initialize with empty dictionary if file does not exist
    try:
        with open(f"{eval_results_dir}/{filename_base}.json", 'r') as f:
            all_metrics = json.load(f)
        print(f"Will append to existing file {eval_results_dir}/{filename_base}.json", end='\n\n')
    except FileNotFoundError:
        print(f"File {eval_results_dir}/{filename_base}.json not found, initializing with empty dictionary.", end='\n\n')
        all_metrics = {}
    for extractor_name in extractor_names:
        img_norms = [args.img_norm] if args.img_norm != 'all' else os.listdir(f"{all_features_save_dir}/{extractor_name}")
        for img_norm in img_norms:
            features_save_dir = f"{all_features_save_dir}/{extractor_name}/{img_norm}"
            for distance_metric in distance_metrics:
                for dimensionality_reduction in dimensionality_reductions:
                    for clustering in clusterings:
                        combo_key = f"{extractor_name}#{img_norm}#{distance_metric}#{dimensionality_reduction}#{clustering}"
                        if (combo_key in all_metrics) and (not args.overwrite):
                            print(f"Skipping {combo_key} - already computed")
                            continue
                        else:
                            print(f"\nComputing {combo_key} ...\n")
                            current_metrics = evaluate_clustering(
                                features_save_dir=features_save_dir,
                                # manual_annotations_dir=args.manual_annotations_dir,
                                manual_annotations_dir=manual_annotations_dir,
                                dimensionality_reduction=dimensionality_reduction,
                                clustering=clustering,
                                distance_metric=distance_metric,
                                verbose=args.verbose,
                            )
                            all_metrics[combo_key] = current_metrics

                        # Save intermediate results to a JSON file - rewrites every time
                        with open(f"{eval_results_dir}/{filename_base}.json", 'w') as f:
                            json.dump(all_metrics, f, indent=4)

    # Save final results to a CSV file
    df = pd.DataFrame(all_metrics).T
    df.to_csv(f"{eval_results_dir}/{filename_base}.csv")


if __name__ == "__main__":
    main()
