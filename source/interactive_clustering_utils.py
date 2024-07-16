import os
import csv
import json
import math
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from PIL import Image

from IPython.display import clear_output


# -------------------------------------------------------------------------------------
# Manual accepting and rejecting of pairs: anchor image + comparison image
# -------------------------------------------------------------------------------------

def visualise_kmeans_cluster(cluster_index, labels_2_imgpaths, labels_2_features, centroids, num_examples=5, random_seed=42, save_path=None):
    img_paths = labels_2_imgpaths[cluster_index]
    features = labels_2_features[cluster_index]
    centroid = centroids[cluster_index]
    distances = euclidean_distances([centroid], features).flatten()

    fig, axes = plt.subplots(1, 1 + num_examples, figsize=(18, 6))

    # Plot the centroid image
    centroid_img_path = img_paths[np.argmin(distances)]
    centroid_img_name = os.path.basename(centroid_img_path).split('.')[0]
    centroid_img = Image.open(centroid_img_path)
    axes[0].imshow(centroid_img)
    axes[0].set_title(f"{centroid_img_name}")
    axes[0].axis('off')

    # Select num_examples random indices
    np.random.seed(random_seed)
    random_indices = np.random.choice(
        range(len(img_paths)), num_examples, replace=False)
    # Sort the random indices by their distance from the centroid
    sorted_random_indices = sorted(random_indices, key=lambda x: distances[x])

    # Plot num_examples random images in a sorted order
    for i, idx in enumerate(sorted_random_indices, start=1):
        img_path = img_paths[idx]
        img_name = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"{img_name}")
        axes[i].axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()


def make_dict_serializable_for_json(session_state):
    session_state_serializable = {}
    for key, value in session_state.items():
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        session_state_serializable[key] = value

    for key in session_state.keys():
        if isinstance(key, np.integer):
            session_state_serializable[int(
                key)] = session_state_serializable.pop(key)

    return session_state_serializable


def update_results_processed(results_csv_file_path, results_processed_file_path):
    results_processed = OrderedDict()
    with open(results_csv_file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            anchor_img_path = row['anchor_img_path']
            cluster_label = int(row['cluster_label'])
            comparison_img_path = row['comparison_img_path']
            user_input = row['user_input']

            if anchor_img_path not in results_processed:
                results_processed[anchor_img_path] = {
                    "cluster_index": cluster_label,
                    "belonging_image_paths": [],
                    "non_belonging_image_paths": []
                }

            if user_input == 'y':
                results_processed[anchor_img_path]["belonging_image_paths"].append(
                    comparison_img_path)
            else:
                results_processed[anchor_img_path]["non_belonging_image_paths"].append(
                    comparison_img_path)

    with open(results_processed_file_path, 'w') as f:
        json.dump(results_processed, f, indent=4)


def display_image_pairs(labels_2_imgpaths, labels_2_features, features, centroid_imgpaths, results_csv_file_path, results_processed_file_path, session_state_file_path):

    # Initialize the results file
    if not os.path.exists(results_csv_file_path):
        with open(results_csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['cluster_label', 'anchor_img_path',
                                'comparison_img_path', 'user_input'])

    # Load session state if exists
    if os.path.exists(session_state_file_path):
        with open(session_state_file_path, 'r') as f:
            session_state = json.load(f)
    else:
        session_state = {'cluster_label': 0, 'image_idx': 0,
                         'sorted_indices': [], 'original_img_path': ''}

    # Get the anchor image and other images in the cluster
    cluster_label = session_state['cluster_label']
    anchor_img_path = centroid_imgpaths[cluster_label]
    img_paths = labels_2_imgpaths[cluster_label]
    anchor_feature = features[labels_2_imgpaths[cluster_label].index(anchor_img_path)]
    start_idx = session_state['image_idx']

    if ('original_img_path' not in session_state) or (session_state['original_img_path'] == ""):
        session_state['original_img_path'] = anchor_img_path

    if ('sorted_indices' not in session_state) or (len(session_state['sorted_indices']) == 0):
        # Compute distances from the anchor image to all other images in the cluster
        distances = euclidean_distances(
            [anchor_feature], labels_2_features[cluster_label]).flatten()
        # Sort in ascending order of euclidean distances
        sorted_indices = np.argsort(distances)
        session_state['sorted_indices'] = sorted_indices.tolist()
    else:
        sorted_indices = np.array(session_state['sorted_indices'])

    for idx in range(start_idx, len(sorted_indices)):
        sorted_idx = sorted_indices[idx]
        img_path = img_paths[sorted_idx]

        if (img_path == anchor_img_path):
            continue

        anchor_img = Image.open(anchor_img_path)
        original_img = Image.open(session_state['original_img_path'])
        img = Image.open(img_path)

        # Display the images side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(anchor_img)
        axes[0].set_title("Anchor Image")
        axes[0].axis('off')

        axes[1].imshow(original_img)
        axes[1].set_title("Current Original Image")
        axes[1].axis('off')

        axes[2].imshow(img)
        axes[2].set_title("Comparison Image")
        axes[2].axis('off')

        plt.show()

        # Prompt user for input
        user_input = input(
            "Do these images belong to the same group? (y/n), mark as original (o), or 'q' to quit: ").strip().lower()
        while user_input not in ['y', 'n', 'o', 'q']:
            print("Invalid input. Please enter 'y', 'n', 'o', or 'q' to quit.")
            user_input = input(
                "Do these images belong to the same group? (y/n), mark as original (o), or 'q' to quit: ").strip().lower()

        if user_input == 'q':
            session_state['image_idx'] = idx
            with open(session_state_file_path, 'w') as f:
                json.dump(make_dict_serializable_for_json(session_state), f)
            print("Session saved. You can continue later.")
            clear_output(wait=True)
            return user_input

        if user_input == 'o':
            session_state['original_img_path'] = img_path
            print("Original image updated.")
            # Record the input as 'y' since it belongs to the same cluster
            user_input = 'y'

        # Record the input in results CSV file
        with open(results_csv_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [cluster_label, anchor_img_path, img_path, user_input])

        # Update results_processed.json
        update_results_processed(
            results_csv_file_path, results_processed_file_path)

        # Increment the image index after processing each image
        session_state['image_idx'] = idx + 1

        # Clear the screen
        clear_output(wait=True)

    # Reset image_idx and move to the next cluster
    session_state['image_idx'] = 0
    session_state['sorted_indices'] = []
    session_state['original_img_path'] = ''
    # Move to the next cluster, do not cycle back to 0
    session_state['cluster_label'] = cluster_label + 1
    with open(session_state_file_path, 'w') as f:
        json.dump(make_dict_serializable_for_json(session_state), f)

    print("All images in the cluster processed. Moving to the next cluster.")


# -------------------------------------------------------------------------------------
# visualisation of the cluster results
# -------------------------------------------------------------------------------------


def visualize_cluster_results_processed_file_path(cluster_id, results_processed_file_path):
    with open(results_processed_file_path, 'r') as f:
        results_processed = json.load(f)

    # Find the anchor image path for the given cluster_id
    cluster_data = list(filter(
        lambda x: x[1]["cluster_index"] == cluster_id, results_processed.items()))

    if len(cluster_data) != 1:
        print(
            f"Error: Expected exactly one cluster with ID: {cluster_id}, found {len(cluster_data)}")
        return

    anchor_img_path, details = cluster_data[0]
    belonging_image_paths = details["belonging_image_paths"]
    non_belonging_image_paths = details["non_belonging_image_paths"]

    # Plot the anchor image
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    anchor_img = Image.open(anchor_img_path)
    plt.imshow(anchor_img)
    plt.title("Anchor Image")
    plt.axis('off')

    # Plot the images in the cluster
    if len(belonging_image_paths) == 0:
        print(f"No images in the cluster with ID: {cluster_id}")
    else:
        num_images = len(belonging_image_paths)
        num_columns = 5
        num_rows = math.ceil(num_images / num_columns)

        plt.figure(figsize=(15, 3 * num_rows))
        for i, img_path in enumerate(belonging_image_paths):
            img = Image.open(img_path)
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(img)
            plt.title(f"Image {i}")
            plt.axis('off')
        plt.suptitle(
            f"{num_images} Images in the Cluster {cluster_id}", fontsize=20)
        plt.show()

    # Plot the images not in the cluster
    if len(non_belonging_image_paths) == 0:
        print(
            f"No non_belonging_image_paths in the cluster with ID: {cluster_id}")
    else:
        num_images = len(non_belonging_image_paths)
        num_columns = 5
        num_rows = math.ceil(num_images / num_columns)

        plt.figure(figsize=(15, 3 * num_rows))
        for i, img_path in enumerate(non_belonging_image_paths):
            img = Image.open(img_path)
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(img)
            plt.title(f"Non-cluster Image {i}")
            plt.axis('off')
        plt.suptitle(
            f"{num_images} Images Not in Cluster {cluster_id}", fontsize=20)
        plt.show()

# -------------------------------------------------------------------------------------
# Clustering, reviewing, and purifying the rejected clusters
# -------------------------------------------------------------------------------------


def visualize_cluster(cluster_label, images):
    print(f"Group {cluster_label}: {len(images)} images")
    num_images = len(images)
    num_columns = 5
    num_rows = math.ceil(num_images / num_columns)

    plt.figure(figsize=(15, 3 * num_rows))
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.suptitle(
        f"Cluster {cluster_label}", fontsize=20)
    plt.show()


def visualize_all_clusters(clusters):
    for label in sorted(clusters.keys()):
        images = clusters[label]
        visualize_cluster(label, images)


def review_single_cluster(cluster_label, images, next_cluster_id):
    remaining_images = images
    non_belonging_images_all = []
    new_clusters = []

    while True:
        visualize_cluster(cluster_label, remaining_images)

        # Ask if the cluster is pure
        while True:
            is_pure = input(
                f"Is Cluster {cluster_label} pure? (y/n/q to quit): ").strip().lower()
            if is_pure in ['y', 'n', 'q']:
                break
            print("Invalid input. Please enter 'y', 'n', or 'q' to quit.")

        if is_pure == 'q':
            clear_output(wait=True)
            print("Process interrupted. Nothing will be saved or returned.")
            return None, None, None

        if is_pure == 'y':
            # Clear the screen after processing the cluster
            clear_output(wait=True)
            break  # The cluster is pure, exit the loop

        # If not pure, get the indices of images that do not belong
        while True:
            non_belonging_indices = input(
                "Enter ALL indices of images from ONE PURE cluster (comma-separated, starting from 0): ").strip()
            try:
                if non_belonging_indices:
                    non_belonging_indices = list(
                        map(int, non_belonging_indices.split(',')))
                break
            except ValueError:
                print(
                    "Invalid input. Please enter comma-separated indices starting from 0.")

        non_belonging_images = [remaining_images[i]
                                for i in non_belonging_indices]
        non_belonging_images_all.extend(non_belonging_images)
        remaining_images = [img for i, img in enumerate(
            remaining_images) if i not in non_belonging_indices]

        new_clusters.append((next_cluster_id, non_belonging_images))
        next_cluster_id += 1

        print(
            f"Non-belonging images in Cluster {cluster_label}: {non_belonging_images}")
        # Clear the screen after processing the cluster
        clear_output(wait=True)

    # Return remaining images, all non-belonging images grouped by new cluster IDs, and the updated next_cluster_id
    return remaining_images, new_clusters, next_cluster_id


def review_clusters(grouped_images):
    pure_clusters = {}
    # Start new cluster IDs from the next available ID
    next_cluster_id = max(grouped_images.keys()) + 1

    for label in sorted(grouped_images.keys()):
        images = grouped_images[label]
        remaining_images, new_clusters, next_cluster_id = review_single_cluster(
            label, images, next_cluster_id)

        if remaining_images is None:
            return None  # Process was interrupted

        pure_clusters[label] = remaining_images

        # Add new clusters to pure_clusters
        for new_label, new_images in new_clusters:
            pure_clusters[new_label] = new_images

    return pure_clusters


def kmeans_and_review(non_belonging_features, n_clusters, non_belonging_images_paths, pure_rejected_clusters_json_path):
    kmeans_nonbelonging = KMeans(n_clusters=n_clusters, random_state=42)
    non_belonging_labels = kmeans_nonbelonging.fit_predict(
        non_belonging_features)

    # Group the non-belonging images by their KMeans labels
    grouped_non_belonging_images = {}
    for img, label in zip(non_belonging_images_paths, non_belonging_labels):
        if label not in grouped_non_belonging_images:
            grouped_non_belonging_images[label] = []
        grouped_non_belonging_images[label].append(img)

    # Review the groups
    pure_clusters = review_clusters(grouped_non_belonging_images)
    if pure_clusters is None:
        print("Process interrupted. Nothing will be saved or returned.")
        return None

    with open(pure_rejected_clusters_json_path, 'w') as f:
        json.dump(make_dict_serializable_for_json(pure_clusters), f, indent=4)
    return pure_clusters

# -------------------------------------------------------------------------------------
# Merging pure accepted and pure rejected clusters
# -------------------------------------------------------------------------------------


def compute_cluster_pairs(clusters, features, img_paths_2_int_ids, linkage='average'):
    """
    Function to compute pairwise distances based on the linkage method
    """
    cluster_ids = list(clusters.keys())
    pairs = []

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cluster1 = cluster_ids[i]
            cluster2 = cluster_ids[j]

            # Get the image indices for the two clusters
            indices1 = [img_paths_2_int_ids[img_path]
                        for img_path in clusters[cluster1]]
            indices2 = [img_paths_2_int_ids[img_path]
                        for img_path in clusters[cluster2]]

            # Compute pairwise distances between all images in the two clusters
            distances = euclidean_distances(
                features[indices1], features[indices2])

            if linkage == 'single':
                # Find the minimum distance (closest pair)
                distance = np.min(distances)
            elif linkage == 'complete':
                # Find the maximum distance (farthest pair)
                distance = np.max(distances)
            elif linkage == 'average':
                # Find the average distance
                distance = np.mean(distances)
            else:
                raise ValueError(
                    "Invalid linkage parameter. Choose from ['complete', 'average', 'single']")

            pairs.append((cluster1, cluster2, distance))

    pairs.sort(key=lambda x: x[2])
    return pairs


def recompute_cluster_distances(clusters, features, img_paths_2_int_ids, cluster_id, linkage='average'):
    """
    Function to recompute distances involving the a cluster, don't recompute all pairs
    """
    cluster_ids = list(clusters.keys())
    pairs = []

    for other_cluster in cluster_ids:
        if other_cluster == cluster_id:
            continue

        # Get the image indices for the two clusters
        indices1 = [img_paths_2_int_ids[img_path]
                    for img_path in clusters[cluster_id]]
        indices2 = [img_paths_2_int_ids[img_path]
                    for img_path in clusters[other_cluster]]

        # Compute pairwise distances between all images in the two clusters
        distances = euclidean_distances(features[indices1], features[indices2])

        if linkage == 'single':
            # Find the minimum distance (closest pair)
            distance = np.min(distances)
        elif linkage == 'complete':
            # Find the maximum distance (farthest pair)
            distance = np.max(distances)
        elif linkage == 'average':
            # Find the average distance
            distance = np.mean(distances)
        else:
            raise ValueError(
                "Invalid linkage parameter. Choose from ['complete', 'average', 'single']")

        pairs.append((cluster_id, other_cluster, distance))

    return pairs


def merge_clusters_interactively(clusters, features, img_paths_2_int_ids, max_num_clusters, linkage, patience):
    cluster_ids = set(clusters.keys())
    seen_pairs = set()
    patience_counter = 0

    sorted_pairs = compute_cluster_pairs(
        clusters=clusters, features=features, img_paths_2_int_ids=img_paths_2_int_ids, linkage=linkage)

    while (len(cluster_ids) > max_num_clusters) or (patience_counter < patience):
        for (cluster1, cluster2, dist) in sorted_pairs:
            if cluster1 not in cluster_ids or cluster2 not in cluster_ids:
                continue
            if (cluster1, cluster2) in seen_pairs or (cluster2, cluster1) in seen_pairs:
                continue

            print("Clusters left:", len(cluster_ids))
            print(
                f"Clusters left > max_num_clusters: {len(cluster_ids) > max_num_clusters}")
            print("patience_counter:", patience_counter)
            print(
                f"patience_counter < patience: {patience_counter < patience}")
            print(
                f"While condition satisfied: {(len(cluster_ids) > max_num_clusters) or (patience_counter < patience)}")
            print()
            # Visualize clusters
            visualize_cluster(cluster1, clusters[cluster1])
            visualize_cluster(cluster2, clusters[cluster2])

            # Ask user for input
            while True:
                merge_decision = input(
                    f"Do you want to merge Cluster {cluster1} and Cluster {cluster2}? (y/n/q to quit): ").strip().lower()
                if merge_decision in ['y', 'n', 'q']:
                    break
                print("Invalid input. Please enter 'y', 'n', or 'q' to quit.")

            if merge_decision == 'q':
                clear_output(wait=True)
                print("Process interrupted. Dictionary of clusters, sorted_pairs, seen_pairs will be returned.")
                return {
                    "clusters": clusters,
                    "sorted_pairs": sorted_pairs,
                    "seen_pairs": seen_pairs,
                }
            elif merge_decision == 'y':
                clusters[cluster1].extend(clusters[cluster2])
                del clusters[cluster2]
                cluster_ids.remove(cluster2)
                # Update sorted pairs
                # Update sorted pairs by recomputing distances only involving the merged cluster
                sorted_pairs = [
                    (c1, c2, d) for c1, c2, d in sorted_pairs if c1 != cluster2 and c2 != cluster2]
                sorted_pairs.extend(recompute_cluster_distances(
                    clusters, features, img_paths_2_int_ids, cluster1, linkage))
                sorted_pairs.sort(key=lambda x: x[2])
                # Remove any pairs involving cluster1 or cluster2 from seen_pairs
                seen_pairs = {(c1, c2) for c1, c2 in seen_pairs if
                              c1 != cluster1
                              and c2 != cluster1
                              and c1 != cluster2
                              and c2 != cluster2}
                # reset patience_counter
                patience_counter = 0
                clear_output(wait=True)
                break
            elif merge_decision == 'n':
                seen_pairs.add((cluster1, cluster2))
                patience_counter += 1

                if (len(cluster_ids) <= max_num_clusters) and (patience_counter >= patience):
                    print(
                        f"Patience limit {patience} reached while the numer of clusters = {len(cluster_ids)} <= {max_num_clusters} = max_num_clusters. Exiting.")
                    print()
                    break
                clear_output(wait=True)

    return {
        "clusters": clusters,
        "sorted_pairs": sorted_pairs,
        "seen_pairs": seen_pairs,
    }
