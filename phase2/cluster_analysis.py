import os
import sys
import pickle

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from feature_extraction import DEVICE


# Select epoch for analysis
if len(sys.argv) != 2:
    print("Usage: python cluster_analysis.py <epoch>")
    exit(0)
epoch = int(sys.argv[1])
MODEL_FOLDER = os.path.join(
    "output/multilingual/model-bert-xlm/checkpoints", str(epoch)
)
print(f"Analysing {MODEL_FOLDER}")

EXPERIMENT_FOLDER = os.path.join(MODEL_FOLDER, "experiment")
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)

FEATURES_FILEPATH = os.path.join(EXPERIMENT_FOLDER, "extracted_features.pt")

# Load features, kmeans model from experiment folder

# For 1b use
columns = [
    "cluster",
    "eng_sent",
    "emj_sent",
    "pos",
    "dist_to_cluster_0",
    "dist_to_cluster_1",
    "dist_to_cluster_2",
    "dist_to_cluster_3",
]
dframe = {col: list() for col in columns}


dataset = torch.load(FEATURES_FILEPATH, DEVICE)
X = []
labels = []
for data in dataset:
    for pos, embedding in enumerate(data["embeddings"]):
        X.append(embedding)
        eng_sent = data["english_sent"]
        emj_sent = data["emoji_sent"]
        labels.append(f"sent: {eng_sent}, emoji: {emj_sent}, pos: {pos}")

        # For 1b use
        dframe["eng_sent"].append(eng_sent)
        dframe["emj_sent"].append(emj_sent)
        dframe["pos"].append(pos)
X = torch.stack(X)

kmeans_model_file = os.path.join(EXPERIMENT_FOLDER, "kmeans-4.pt")
with open(kmeans_model_file, "rb") as f:
    kmeans = pickle.load(f)

# A list of analysis result in string type
analysis = []

# Analysis 1a: Find k closest points to cluster centers
top_k = 50
pw_dists = pairwise_distances(kmeans.cluster_centers_, X)
for cluster_id, cluster_dists in enumerate(pw_dists):
    analysis.append(f"{top_k} closest data in Cluster {cluster_id}")
    sorted_dists = sorted([(d, i) for i, d in enumerate(cluster_dists)])
    top_sentences = [f"{labels[i]}" for _, i in sorted_dists[:top_k]]
    analysis.append("\n".join(top_sentences))
    analysis.append("")

# Analysis 1b: generate df
pw_dists = pairwise_distances(X, kmeans.cluster_centers_)
for dist_to_clusters in pw_dists:
    cluster = np.argmin(dist_to_clusters)

    dframe["cluster"].append(cluster)
    for i, dist in enumerate(dist_to_clusters):
        dframe[f"dist_to_cluster_{i}"].append(dist)

# Testing 1b
# for i in range(len(dframe["cluster"])):
#     eng_sent = dframe["eng_sent"][i]
#     emj_sent = dframe["emj_sent"][i]
#     pos = dframe["pos"][i]
#     assert labels[i] == f"sent: {eng_sent}, emoji: {emj_sent}, pos: {pos}"
#     assert dframe["cluster"][i] == kmeans.predict(X[i].reshape(1, -1))[0]
# print("tested ok")

DF_FILEPATH = os.path.join(EXPERIMENT_FOLDER, "cluster_analysis_full.csv")
df = pd.DataFrame.from_dict(dframe).to_csv(DF_FILEPATH)

def stringifyFreqTable(eng_sent, emj_sent, freqTable):
    result = []
    result.append(f"Cluster distribution of sent: {eng_sent}")
    result.append(f"                              {emj_sent}")

    result.append(f"Cluster: 0    1    2    3")

    freqStr = "    ".join(map(str, freqTable))
    result.append(f"Count:   {freqStr}")

    totalFreq = sum(freqTable)
    percentStr = "%  ".join([str(freq / totalFreq * 100) for freq in freqTable])
    result.append(f"Percent: {percentStr}")

    result.append("")
    return "\n".join(result)


# Analysis 2: Distribution of cluster in each sent
for i, data in enumerate(dataset[1:]):
    eng_sent = data["english_sent"]
    emj_sent = data["emoji_sent"]
    freqTable = [0 for _ in range(4)]
    for x in data["embeddings"]:
        cluster = kmeans.predict(x.reshape(1, -1))[0]
        freqTable[cluster] += 1
    if len(data["embeddings"]) not in freqTable:
        analysis.append(stringifyFreqTable(eng_sent, emj_sent, freqTable))

# Save analysis
OUTPUT_PATH = os.path.join(EXPERIMENT_FOLDER, "cluster-analysis.txt")
with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(analysis))
print(f"Saved output at {OUTPUT_PATH}")
