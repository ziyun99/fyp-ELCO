import os

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from feature_extraction import DEVICE, EXPERIMENT_FOLDER, FEATURES_FILEPATH

RAND_SEED = 27


def load_features(filepath):
    """
    Load feature embeddings from filepath into device.
    """
    dataset = torch.load(filepath, DEVICE)

    features = []
    labels = []
    num_em = 0
    for data in dataset:
        for embedding in data["embeddings"]:
            features.append(embedding)
            labels.append(data["english_sent"])
        num_em += data["num_emoji"] + 1
    features = torch.stack(features)

    assert len(features) == num_em
    assert len(features) == len(labels)
    assert len(features[0]) == 768

    print(f"Loaded features: {filepath} as a Tensor: {features.shape}")
    return features


def cluster_features(features, n_clusters=6):
    """
    Cluster features with KMeans model, generate a jpg and returns the
    KMeans inertia value.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RAND_SEED)
    kmeans.fit(features)
    inertia = kmeans.inertia_

    print(f"Fitted features into KMeans model: {kmeans}, inertia = {inertia}")

    # Visualization
    pca = PCA(random_state=RAND_SEED)
    reduced_features = pca.fit_transform(features)
    reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

    plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1], c=kmeans.predict(features)
    )
    plt.scatter(
        reduced_cluster_centers[:, 0],
        reduced_cluster_centers[:, 1],
        marker="x",
        s=150,
        c="b",
    )

    PLOT_FILEPATH = os.path.join(EXPERIMENT_FOLDER, f"cluster-{n_clusters}.jpg")
    plt.savefig(PLOT_FILEPATH)

    print(f"Generated graph: {PLOT_FILEPATH}")
    return inertia


def plot_inertia(inertias, N):
    """
    Plot the n_clusters-to-inertia graph for elbow method
    """
    plt.figure()
    plt.plot(N, inertias, "bx-")
    plt.xlabel("Clusters")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal n")

    ELBOW_FILEPATH = os.path.join(EXPERIMENT_FOLDER, "elbow-method.jpg")
    plt.savefig(ELBOW_FILEPATH)


def main():
    features = load_features(FEATURES_FILEPATH)

    inertias = []
    N = range(2, 11)
    for n in N:
        inertias.append(cluster_features(features, n))

    plot_inertia(inertias, N)


if __name__ == "__main__":
    main()
