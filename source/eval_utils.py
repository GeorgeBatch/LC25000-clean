from sklearn.metrics import pairwise_distances
from copy import deepcopy

import numpy as np

import umap
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import confusion_matrix
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

from source.constants import RANDOM_SEED


def reduce_feature_dimensionality(features, method='NoReduction'):
    if method == 'NoReduction':
        features_reduced = deepcopy(features)
    elif method.startswith('PCA'):
        n_components = float(method.split('-')[1])
        if n_components > 1:
            n_components = int(n_components)
        reduction_algo = PCA(n_components=n_components, random_state=RANDOM_SEED)
        features_reduced = reduction_algo.fit_transform(features)
    elif method.startswith('UMAP'):
        n_components = int(method.split('-')[1])
        reduction_algo = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED)
        features_reduced = reduction_algo.fit_transform(features)
    else:
        raise NotImplementedError

    return features_reduced


def get_clustering_centroids(features, cluster_labels):
    """
    Get the centroids of the clusters for a given dataset.

    Parameters:
    features (array-like): The input dataset.
    cluster_labels (array-like): The cluster labels for the input dataset.

    Returns:
    array-like: The centroids of the clusters.
    """
    if isinstance(cluster_labels, list):
        cluster_labels = np.array(cluster_labels)
    unique_cluster_labels = np.unique(cluster_labels)
    cluster_centroids = np.zeros((len(unique_cluster_labels), features.shape[1]))

    # assumes that unique_cluster_labels are in the range[0, n_clusters-1]
    assert set(unique_cluster_labels) == set(range(len(unique_cluster_labels))), "Cluster labels should be in the range[0, n_clusters-1]"
    assert np.all(unique_cluster_labels == np.arange(len(unique_cluster_labels))), "unique_cluster_labels should be a sorted array of integers from 0 to n_clusters-1"
    for cluster_label in unique_cluster_labels:
        cluster_centroids[cluster_label] = np.mean(
            features[cluster_labels == cluster_label],
            axis=0)

    # convert cluster_centroids to np.float32 to be consistent with sklearn.cluster.KMeans.cluster_centers_
    cluster_centroids = cluster_centroids.astype(np.float32)

    return cluster_centroids


def get_clustering_labels(features, n_clusters, method, random_state, return_model=False):
    """
    Get the clustering labels for a given dataset using different clustering methods.

    Parameters:
    features (array-like): The input dataset.
    n_clusters (int): The number of clusters to generate.
    method (str, optional): The clustering method to use. Defaults to 'kmeans'.
    return_model (bool, optional): Whether to return the clustering algorithm object. Defaults to False.

    Returns:
    array-like: The cluster labels for the input dataset.
    object: The clustering algorithm object. Only returned if return_model is True.

    Raises:
    NotImplementedError: If the specified clustering method is not implemented.
    """
    if method == 'kmeans':
        clustering_algo = KMeans(
            n_clusters=n_clusters, random_state=random_state)
    elif method == 'agglomerative-single':
        clustering_algo = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    elif method == 'agglomerative-average':
        clustering_algo = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    elif method == 'agglomerative-complete':
        clustering_algo = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    else:
        raise NotImplementedError("Clustering method not implemented")

    predicted_cluster_labels = clustering_algo.fit_predict(features)

    if return_model:
        return predicted_cluster_labels, clustering_algo
    else:
        return predicted_cluster_labels


def compute_connectivity_matrix(clusters_dict, imgpaths_2_intids):
    """
    Computes the connectivity matrix based on the given clusters and image paths to integer IDs mapping.

    Args:
        clusters_dict (dict): A dictionary containing cluster IDs as keys and lists of image paths as values.
        imgpaths_2_intids (dict): A dictionary mapping image paths to their corresponding integer IDs in the features array.

    Returns:
        numpy.ndarray: The computed connectivity matrix.

    Raises:
        AssertionError: If any diagonal element of the connectivity matrix is non-zero.

    """
    connectivity_matrix = np.zeros(
        (len(imgpaths_2_intids), len(imgpaths_2_intids)))
    for cluster_id, image_list in clusters_dict.items():
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                connectivity_matrix[imgpaths_2_intids[image_list[i]],
                                    imgpaths_2_intids[image_list[j]]] = 1
                connectivity_matrix[imgpaths_2_intids[image_list[j]],
                                    imgpaths_2_intids[image_list[i]]] = 1

    assert all(
        [connectivity_matrix[i, i] == 0
         for i in range(len(connectivity_matrix))]
    ), "Diagonal elements should all be zero for the true connectivity matrix."

    return connectivity_matrix


def compute_clustering_metrics(true_connectivity_vector, predicted_connectivity_vector, true_cluster_labels, predicted_cluster_labels):
    """
    Compute various clustering evaluation metrics.

    Parameters:
    true_connectivity_vector (array-like): The true connectivity vector.
    predicted_connectivity_vector (array-like): The predicted connectivity vector.
    true_cluster_labels (array-like): The true cluster labels.
    predicted_cluster_labels (array-like): The predicted cluster labels.

    Returns:
    dict: A dictionary containing the computed clustering metrics:
        - Confusion Matrix
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Fowlkes-Mallows Index
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        - Homogeneity
        - Completeness
        - V-Measure
    """
    # ------------------------------------------------------------------------
    # Connectivity Metrics: take binary connectivity vectors
    # ------------------------------------------------------------------------
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(
        true_connectivity_vector, predicted_connectivity_vector).ravel()
    # Convert numpy.int64 to int
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Precision / Positive Predictive Value (PPV)
    precision = tp / (tp + fp)
    # Recall / Sensitivity / True Positive Rate (TPR)
    recall = tp / (tp + fn)
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)

    # # Specificity / True Negative Rate (TNR)
    specificity = tn / (tn + fp)
    # # balanced accuracy = (specificity + sensitivity) / 2 = (TPR + TNR) / 2
    balanced_accuracy = (specificity + recall) / 2

    # ------------------------------------------------------------------------
    # Clustering Metrics: take cluster labels
    # ------------------------------------------------------------------------
    fmi = fowlkes_mallows_score(true_cluster_labels, predicted_cluster_labels)
    ari = adjusted_rand_score(true_cluster_labels, predicted_cluster_labels)
    nmi = normalized_mutual_info_score(
        true_cluster_labels, predicted_cluster_labels)
    homogeneity = homogeneity_score(
        true_cluster_labels, predicted_cluster_labels)
    completeness = completeness_score(
        true_cluster_labels, predicted_cluster_labels)
    v_measure = v_measure_score(true_cluster_labels, predicted_cluster_labels)

    metrics = {
        # Connectivity Metrics
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy,
        # Clustering Metrics
        "Fowlkes-Mallows Index": fmi,
        "Adjusted Rand Index (ARI)": ari,
        "Normalized Mutual Information (NMI)": nmi,
        "Homogeneity": homogeneity,
        "Completeness": completeness,
        "V-Measure": v_measure
    }

    return metrics


def is_closest_connected(i, dist_matrix, true_connectivity_matrix):
    # find closest image index, excluding the same image
    dist_matrix[i, i] = np.inf
    closest_img_index = np.argmin(dist_matrix[i])
    # return if the closest image is connected to the current image
    return true_connectivity_matrix[i, closest_img_index]


def precision_at_1(features, true_connectivity_matrix, metric='euclidean'):
    dist_matrix = pairwise_distances(features, features, metric=metric)
    num_total_images = features.shape[0]

    # could be optimised using vectorised operations, but this is more readable
    connected_to_closest_nbr_arr = np.zeros(num_total_images, dtype=int)
    for i in range(num_total_images):
        connected_to_closest_nbr_arr[i] = is_closest_connected(
            i, dist_matrix, true_connectivity_matrix)

    # connected_to_closest_k_nbrs_arr is already binary, does no change if we take (connected_to_closest_k_nbrs_arr==1)
    precision_at_1 = np.mean(connected_to_closest_nbr_arr)
    return precision_at_1


def get_closest_k_connection_labels(i, dist_matrix, true_connectivity_matrix, k=5):
    # find k closest image indices, excluding the same image
    dist_matrix[i, i] = np.inf
    # argpartition is faster than argsort for large arrays
    closest_img_indices_unsorted = np.argpartition(dist_matrix[i], k)[:k]
    # Sort the closest neighbors up to k - this is not necessary for the precision@k calculation
    #   but is useful if we do not take mean over `k` later
    closest_nbrs_sorted = closest_img_indices_unsorted[np.argsort(
        dist_matrix[i][closest_img_indices_unsorted])]
    # return if any of the k closest images are connected to the current image
    return true_connectivity_matrix[i, closest_nbrs_sorted]


def precision_at_k(features, true_connectivity_matrix, k=5, metric='euclidean', ):
    dist_matrix = pairwise_distances(features, features, metric=metric)
    num_total_images = features.shape[0]

    # could be optimised using vectorised operations, but this is more readable
    connected_to_closest_k_nbrs_arr = np.zeros(
        (num_total_images, k), dtype=int)
    for i in range(num_total_images):
        connected_to_closest_k_nbrs_arr[i] = get_closest_k_connection_labels(
            i, dist_matrix, true_connectivity_matrix, k)

    # connected_to_closest_k_nbrs_arr is already binary, does no change if we take (connected_to_closest_k_nbrs_arr==1)
    precision_at_k = np.mean(connected_to_closest_k_nbrs_arr, axis=(0, 1))
    return precision_at_k
