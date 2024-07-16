import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visual_plot_kmeans(k_means_labels, X_norm, k_means_cluster_centers):
    # Credit: https://colab.research.google.com/drive/1F-hpE_ERmte-5Er89hAMULGZcF4lkYL5#scrollTo=qO1x2Zt7BmDr
    plt.figure(figsize=(6, 4))
    # Colors uses a color map, which will produce an array of colors based on
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # data point is in.
    for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
        my_members = (k_means_labels == k)
        # Define the centroid, or cluster center.
        if len(k_means_cluster_centers) < 2:
            print('no center needed')
        else:
            cluster_center = k_means_cluster_centers[k]
        ax1.plot(X_norm[my_members, 0], X_norm[my_members, 1], 'w', markerfacecolor=col, marker='.', markersize=12)
        if len(k_means_cluster_centers) < 2:
            print('no center needed')
        else:
            ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=15)

    ax2.scatter(X_norm[:, 0], X_norm[:, 1], cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title('KMeans')
    ax2.set_title('Actual clusters')
    ax1.set_xticks(())
    ax1.set_yticks(())
    ax2.set_xticks(())
    ax2.set_yticks(())
    plt.show()


def plot_myReducedDim(X_reduced, y, no_of_labels, method):
    # restructure the data
    df = pd.DataFrame(X_reduced)
    df['categories'] = np.reshape(y, [y.shape[0], 1])
    plt.figure(figsize=(12, 8))

    # you can write above function as just one statement instead of multiple if-elif
    assert method in ['PCA', 'KPCA', 'tsne', 'svd', 'mds', 'isomap']
    method = method.lower()

    df[f'{method}-one'] = X_reduced[:, 0]
    df[f'{method}-two'] = X_reduced[:, 1]
    sns.scatterplot(
        x=f"{method}-one", y=f"{method}-two",
        hue="categories",
        palette=sns.color_palette("hls", no_of_labels),
        data=df.loc[:, :],
        legend="full",
        alpha=0.3)
