import os

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from feature_extraction import DEVICE, EXPERIMENT_FOLDER, FEATURES_FILEPATH

RAND_SEED = 27


def load_features(filepath, device):
    """
    Load feature embeddings from SAVE_FILEPATH
    """
    dataset = torch.load(filepath, device)

    features = []
    num_em = 0
    for data in dataset:
        for embedding in data["embeddings"]:
            features.append(embedding)
        num_em += data["num_emoji"] + 1
    features = torch.stack(features)

    assert len(features) == num_em
    assert len(features[0]) == 768

    print(f"Loaded features: {filepath} as a Tensor: {features.shape}")
    return features


def cluster_features(features, n_clusters=6, random_state=RAND_SEED):
    """
    Fit features into KMeans model for clustering
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(features)

    print(f"Fitted features into KMeans model: {kmeans}")

    """
    Visualization
    """
    pca = PCA(random_state=random_state)
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


def main():
    features = load_features(FEATURES_FILEPATH, DEVICE)

    for i in range(2, 11):
        cluster_features(features, i)


if __name__ == "__main__":
    main()
