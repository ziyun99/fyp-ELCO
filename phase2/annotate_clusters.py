import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from feature_extraction import DEVICE

# Load necessary files, models, etc.
if len(sys.argv) != 2:
    print("Usage: python make_graph.py <epoch>")
    exit(0)
epoch = int(sys.argv[1])
MODEL_FOLDER = os.path.join(
    "output/multilingual/model-bert-xlm/checkpoints", str(epoch)
)
EXPERIMENT_FOLDER = os.path.join(MODEL_FOLDER, "experiment")
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)

KMEANS_MODEL_PATH = os.path.join(EXPERIMENT_FOLDER, "kmeans-4.pt")
with open(KMEANS_MODEL_PATH, "rb") as f:
    kmeans = pickle.load(f)

FEATURES_PATH = os.path.join(EXPERIMENT_FOLDER, "extracted_features.pt")
dataset = torch.load(FEATURES_PATH, DEVICE)
X = []
for data in dataset:
    for embedding in data["embeddings"]:
        X.append(embedding)
X = torch.stack(X)

pca = PCA(n_components=2, random_state=27)
reduced_features = pca.fit_transform(X)
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

colors = np.array(["r", "g", "c", "y", "m"])
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=colors[kmeans.labels_])

# plt.scatter(
#     reduced_cluster_centers[:, 0],
#     reduced_cluster_centers[:, 1],
#     marker="x",
#     s=150,
#     c="b",
# )

# Find k closest points to cluster centers
# data_str = []
# for data in dataset:
#     for pos, embedding in enumerate(data["embeddings"]):
#         eng_sent = data["english_sent"]
#         emj_sent = data["emoji_sent"]
#         data_str.append(f"sent: {eng_sent}, emoji: {emj_sent}, pos: {pos}")

# top_k = 50
# analysis = []
# pw_dists = pairwise_distances(reduced_cluster_centers, reduced_features)
# for cluster_id, cluster_dists in enumerate(pw_dists):
#     analysis.append(f"{top_k} closest data in Cluster {cluster_id}")
#     sorted_dists = sorted([(d, i) for i, d in enumerate(cluster_dists)])
#     top_sentences = [f"({i}) dist: {d} {data_str[i]}" for d, i in sorted_dists[:top_k]]
#     analysis.append("\n".join(top_sentences))
#     analysis.append("")
# with open("analysis.txt", "w") as f:
#     f.write("\n".join(analysis))

trackIds = [11, 69, 400, 520, 999, 1314, 2713, 3693]

trackX = torch.stack([X[i] for i in trackIds])
track_features = pca.transform(trackX)
plt.scatter(track_features[:, 0], track_features[:, 1], c="k")

texts = []
for data in dataset:
    for _ in data["embeddings"]:
        texts.append(data["english_sent"])

trackTexts = [texts[i] for i in trackIds]
# trackTexts = ["woaini", "baobao the best", "luv u", "hehe"]
for i, text in enumerate(trackTexts):
    print(f"{i}: {text}")
    plt.annotate(
        text,
        xy=(track_features[i][0], track_features[i][1]),
        xytext=(5, 2),
        textcoords="offset points",
    )

PLOT_PATH = os.path.join("trace", f"annotated-{epoch}.jpg")
plt.savefig(PLOT_PATH)
print(f"Generated graph: {PLOT_PATH}")
