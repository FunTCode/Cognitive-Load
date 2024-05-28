import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

def detect_outliers_knn(X, labels, threshold_percentile=90, k=3):
    """
    The K-nearest neighbor method is used to detect outliers.

    Parameters:
    -X: indicates the input data set.
    - labels: indicates a label for a data set.
    - threshold_percentile: specifies the threshold percentile for estimating outliers. The default value is 90%.
    -k: indicates the number of neighbors. The default value is 3.

    Return value:
    - outliers_mask: indicates the Boolean mask of the data point marked as an outlier.
    """
    outliers_mask = np.zeros(len(X), dtype=bool)

    for label in np.unique(labels):
        label_indices = (labels == label)
        label_data = X[label_indices]

        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(label_data)
        distances, _ = neighbors.kneighbors(label_data)

        # 计算每个点到第k个邻居的平均距离
        avg_distances = np.mean(distances[:, 1:], axis=1)

        # 设置阈值，大于阈值的点被认为是异常值
        threshold = np.percentile(avg_distances, threshold_percentile)
        outliers_mask[label_indices] = (avg_distances > threshold)

    return outliers_mask

def detect_outliers_2dshow(X, threshold_percentile=95, n_clusters=3, random_state=42, show=False):
    """
    Outliers were detected using t-SNE and k-means clustering.

    Parameters:
    -X: Input data set with shape (n_samples, n_features).
    - threshold_percentile: specifies the threshold percentile for estimating outliers. The default value is 95%.
    -n_clusters: k-means the number of clusters in a cluster. The default is 3.
    - random_state: random number seed. The default value is None.
    - show: Whether to display visual results. The default value is False.

    Return value:
    - outliers: Data points marked as outliers.
    """
    # execute t-SNE
    tsne = TSNE(n_components=2, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # execute k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++')
    kmeans.fit(X_tsne)

    # Gets the cluster label
    labels = kmeans.labels_

    # Calculate the distance of each point to the center of its cluster
    distances = np.min(kmeans.transform(X_tsne), axis=1)

    # Define a threshold to determine which points are considered outliers
    threshold = np.percentile(distances, threshold_percentile)

    # A Boolean array of labeled outliers
    outliers_mask = distances > threshold

    if show:
        # visualization
        plt.figure(figsize=(8, 6))
        # Draw different clustering clusters
        for i in range(n_clusters):
            plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], label=f'Cluster {i + 1}')
        # Labeled outlier
        plt.scatter(X_tsne[outliers_mask, 0], X_tsne[outliers_mask, 1], c='r', marker='x', label='Outliers')
        plt.title('t-SNE with Clustering and Outliers Detection')
        plt.legend()
        plt.show()

    return outliers_mask



def detect_outliers_3dshow(X, threshold_percentile=95, n_clusters=3, random_state=42, show=False):
    """
    Outliers were detected using t-SNE and k-means clustering.

    Parameters:
    -X: Input data set with shape (n_samples, n_features).
    - threshold_percentile: specifies the threshold percentile for estimating outliers. The default value is 95%.
    -n_clusters: k-means the number of clusters in a cluster. The default is 3.
    - random_state: random number seed. The default value is None.
    - show: Whether to display visual results. The default value is False.

    Return value:
    - outliers: Data points marked as outliers.
    """
    # execute t-SNE
    tsne = TSNE(n_components=3, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # execute k-means 
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++')
    kmeans.fit(X_tsne)

    # Calculate the distance of each point to the center of its cluster
    distances = np.min(kmeans.transform(X_tsne), axis=1)

    # Define a threshold to determine which points are considered outliers
    threshold = np.percentile(distances, threshold_percentile)

    # A Boolean array of labeled outliers
    outliers_mask = distances > threshold

    if show:
        # 3D visualization
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # Draw different clustering clusters
        for i in range(n_clusters):
            ax.scatter(X_tsne[kmeans.labels_ == i, 0], X_tsne[kmeans.labels_ == i, 1], X_tsne[kmeans.labels_ == i, 2], label=f'Cluster {i+1}')
        # Plot outliers
        ax.scatter(X_tsne[outliers_mask, 0], X_tsne[outliers_mask, 1], X_tsne[outliers_mask, 2], c='r', marker='x', label='Outliers')
        ax.set_title('t-SNE with Clustering and Outliers Detection')
        ax.legend()
        plt.show()

    return outliers_mask

# # sample
# X, _ = make_blobs(n_samples=1000, n_features=9, centers=3, random_state=42)
# outliers = detect_outliers_2dshow(X,n_clusters=3,show=True)
# outliers = detect_outliers_3dshow(X,n_clusters=3,show=True)

pass