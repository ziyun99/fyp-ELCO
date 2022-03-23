import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from feature_extraction import SAVE_FILEPATH

RAND_SEED = 27


def main():
    """
    Load feature embeddings from SAVE_FILEPATH
    """
    dataset = torch.load(SAVE_FILEPATH)
    features = []
    num_em = 0
    for data in dataset:
        for embedding in data["embeddings"]:
            features.append(embedding)
        num_em += data["num_emoji"] + 1

    assert len(features) == num_em
    assert len(features[0]) == 768

    """
    Fit features into KMeans model for clustering
    """
    kmeans = KMeans(n_clusters=5, random_state=RAND_SEED)
    kmeans.fit(features)

    """
    Visualization
    """
    pca = PCA(n_components=2, random_state=RAND_SEED)
    reduced_features = pca.fit_transform(features.toarray())
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
    


if __name__ == "__main__":
    main()
