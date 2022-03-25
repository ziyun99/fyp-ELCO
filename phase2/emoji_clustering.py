import logging
import os
import pickle

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from feature_extraction import DEVICE, EXPERIMENT_FOLDER, FEATURES_FILEPATH

RAND_SEED = 27

LOG_FILEPATH = os.path.join(EXPERIMENT_FOLDER, "logfile")
logging.basicConfig(
    filename=LOG_FILEPATH, filemode="w", format="%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def visualizePCA(n, kmeans, features, labels, filename="cluster"):
    """
    Plot large dimension features into 2D space
    """
    pca = PCA(random_state=RAND_SEED)
    reduced_features = pca.fit_transform(features)
    reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_)
    plt.scatter(
        reduced_cluster_centers[:, 0],
        reduced_cluster_centers[:, 1],
        marker="x",
        s=150,
        c="b",
    )

    # label each EM point with english sentence
    for i, label in enumerate(labels):
        plt.annotate(label,
                        xy=(reduced_features[i][0], reduced_features[i][1]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')

    PLOT_FILEPATH = os.path.join(EXPERIMENT_FOLDER, f"{filename}.jpg")
    plt.savefig(PLOT_FILEPATH)

    logging.info(f"Generated graph: {PLOT_FILEPATH}")

    # save pca model
    pca_model_file = os.path.join(EXPERIMENT_FOLDER, "pca-{}.pt".format(n))
    pickle.dump(pca, open(pca_model_file, "wb"))

def augmentClusters(dataset, labels):
    """
    Augment the dataset with the predicted clusters.
    """
    i = 0
    for data in dataset:
        clusters = []
        for _ in data["embeddings"]:
            clusters.append(labels[i])
            i += 1
        data["clusters"] = clusters
    assert i == len(labels)

    logging.info("Succesfully augmented cluster information into dataset")


def evaluateLabels(dataset):
    """
    Evaluate how many % of sentences where all [EM] are labelled together.
    """
    total = len(dataset)
    correct = 0
    for data in dataset:
        clusters = data["clusters"]
        all_correct = all(cluster == clusters[0] for cluster in clusters)
        # TODO: cosine_score = calculate_blablabla(clusters)
        if all_correct:
            correct += 1
        else:
            english_sent = data["english_sent"]
            emoji_sent = data["emoji_sent"]
            logging.info(
                f"=> Label mismatch:\n{english_sent}\n{emoji_sent}\n{clusters}\n"
            )

    logging.info(f"Accuracy: {correct / total}")


def plot_elbow_method(x_value, y_value, title, x_label, y_label, filename):
    """
    Plot helper
    """
    plt.figure()
    plt.plot(x_value, y_value)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    FILEPATH = os.path.join(EXPERIMENT_FOLDER, f"{filename}.jpg")
    plt.savefig(FILEPATH)


def main():
    """
    Load feature embeddings from filepath into device.
    """
    dataset = torch.load(FEATURES_FILEPATH, DEVICE)

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
    logging.info(f"Loaded features: {FEATURES_FILEPATH} as a Tensor: {features.shape}")

    """
    Feature clustering using 2 to 10 clusters
    """
    inertias = []
    N = range(2, 11)
    for n in N:
        logging.info(f"Begin clustering with {n}-clusters")
        kmeans = KMeans(n_clusters=n, random_state=RAND_SEED)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        logging.info(f"Fitted features into KMeans model: {kmeans}")

        augmentClusters(dataset, kmeans.labels_)
        evaluateLabels(dataset)
        visualizePCA(n, kmeans, features, labels, f"cluster-{n}")

        # save kmeans model
        kmeans_model_file = os.path.join(EXPERIMENT_FOLDER, "kmeans-{}.pt".format(n))
        pickle.dump(kmeans, open(kmeans_model_file, "wb"))


    plot_elbow_method(
        x_value=N,
        y_value=inertias,
        title="Elbow method",
        x_label="n_clusters",
        y_label="inertias",
        filename="elbow-method",
    )


if __name__ == "__main__":
    main()
