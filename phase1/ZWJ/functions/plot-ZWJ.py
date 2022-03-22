import pandas as pd
import re
import json 
import emoji
import functools
import operator
import statistics as st
import time
from sentence_transformers import SentenceTransformer, util
import numpy as np

datasets= ["ZWJ", "AN"]
dataset_idx = 0 
dataset_name = datasets[dataset_idx]

def read_score_df():
    f = open("data/ZWJ/zwj-scoring-mpnet.json")
    data_dict = json.load(f)

    scores = []
    scores_ranked = []
    for concept in data_dict:
        scores.append(data_dict[concept]["scores"])
        scores_ranked.append(data_dict[concept]['scores_ranked'])

    concepts = list(data_dict.keys())
    concept_count = len(concepts)

    scores = np.array(scores)
    augment_id = 0
    randneg = scores[:, augment_id, 0]
    baseline = scores[:, augment_id, 1]
    pos = scores[:, augment_id, 2]

    augment_id = 1
    augment_randneg = scores[:, augment_id, 0]
    augment_baseline = scores[:, augment_id, 1]
    augment_pos = scores[:, augment_id, 2]

    score_df = pd.DataFrame()
    score_df['concept'] = concepts
    score_df['randneg'] = randneg
    score_df['baseline'] = baseline
    score_df['pos'] = pos
    score_df['augment_randneg'] = augment_randneg
    score_df['augment_baseline'] = augment_baseline
    score_df['augment_pos'] = augment_pos

    print(score_df.head())
    return score_df

# score_df = read_score_df()
# score_df.to_csv("scores/zwj-score.csv")
# score_df.to_excel("scores/zwj-score.csv")

import plotly.express as px
import plotly.graph_objects as go

def plot_scores(score_df):
    columns = score_df.columns.values.tolist()
    columns.remove('concept')

    for col in columns:
        img_path = "figures/{}/{}_{}.png".format(dataset_name, dataset_name, col)
        fig = px.bar(score_df,  y=[col])
        fig.write_image(img_path, format="png", width=600, height=350, scale=2)

def plot_line_graph(score_df, col, img_name):
    img_path = "figures/{}/{}_{}.png".format(dataset_name, dataset_name, img_name)
    fig = px.line(score_df,  y=col)
    fig.update_layout(xaxis_title="ZWJ concepts", yaxis_title="similarity scores", legend_title="Emoji sequences", margin = {'l':5,'r':5,'t':5,'b':5},)
    fig.write_image(img_path, format="png", width=600, height=350, scale=2)

# score_df = read_score_df()
# col = ['pos', 'augment_pos']
# img_name = "line_graph_pos"
# plot_line_graph(score_df, col, img_name)

# plot_scores(score_df)



def compute_mean_scores(score_df):
    randneg = [np.array(score_df['randneg'])]
    baseline = [np.array(score_df['baseline'])]
    pos = [np.array(score_df['pos'])]

    augment_randneg = [np.array(score_df['augment_randneg'])]
    augment_baseline = [np.array(score_df['augment_baseline'])]
    augment_pos = [np.array(score_df['augment_pos'])]

    randneg_mean = np.mean(randneg)
    baseline_mean = np.mean(baseline)
    pos_mean = np.mean(pos)
    augment_randneg_mean = np.mean(augment_randneg)
    augment_baseline_mean = np.mean(augment_baseline)
    augment_pos_mean = np.mean(augment_pos)

    mean = [randneg_mean, baseline_mean, pos_mean, augment_randneg_mean, augment_baseline_mean, augment_pos_mean]
    mean = [round(m, 4) for m in mean]
    print(mean)

score_df = read_score_df()
compute_mean_scores(score_df)
mean = score_df.mean()
print(mean)

from sklearn.metrics import ndcg_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score

def run_metrics(score_df):
    metrics_df = score_df
    randneg = [np.array(score_df['randneg'])]
    baseline = [np.array(score_df['baseline'])]
    pos = [np.array(score_df['pos'])]

    scores = np.concatenate((randneg, baseline, pos), axis=0).T
    print(scores.shape)

    true_relevance = np.asarray([[-1, 0, 1]])
    ndcg_scores = []
    for score in scores:
        score = np.asarray([score])
        ndcg = ndcg_score(true_relevance, score)
        # print(true_relevance, score, ndcg)
        ndcg_scores.append(round(ndcg, 4))

    ndcg = np.mean(ndcg_scores)
    print(ndcg)
    metrics_df["ndcg_score"] = ndcg_scores
    
    true_relevance = np.asarray([[0, 0, 1]])
    ranking_losses = []
    for score in scores:
        score = np.asarray([score])
        ranking_loss = label_ranking_loss(true_relevance, score)
        # print(true_relevance, score, ranking_loss)
        ranking_losses.append(round(ranking_loss, 4))

    ranking_loss = np.mean(ranking_losses)
    print(ranking_loss)
    metrics_df["ranking_loss"] = ranking_losses

    true_relevance = np.asarray([[0, 0, 1]])
    map_scores = []
    for score in scores:
        score = np.asarray([score])
        map_score = label_ranking_average_precision_score(true_relevance, score)
        # print(true_relevance, score, map_score)
        map_scores.append(round(map_score, 4))

    map_score = np.mean(map_scores)
    print(map_score)
    metrics_df["map_score"] = map_scores


    metrics_df = score_df
    augment_randneg = [np.array(score_df['augment_randneg'])]
    augment_baseline = [np.array(score_df['augment_baseline'])]
    augment_pos = [np.array(score_df['augment_pos'])]

    augment_scores = np.concatenate((augment_randneg, augment_baseline, augment_pos), axis=0).T
    print(augment_scores.shape)

    augment_true_relevance = np.asarray([[-1, 0, 1]])
    augment_ndcg_scores = []
    for score in augment_scores:
        score = np.asarray([score])
        ndcg = ndcg_score(true_relevance, score)
        # print(augment_true_relevance, score, ndcg)
        augment_ndcg_scores.append(round(ndcg, 4))

    augment_ndcg = np.mean(augment_ndcg_scores)
    print(augment_ndcg)
    metrics_df["augment_ndcg_score"] = augment_ndcg_scores
    
    augment_true_relevance = np.asarray([[0, 0, 1]])
    augment_ranking_losses = []
    for score in augment_scores:
        score = np.asarray([score])
        ranking_loss = label_ranking_loss(augment_true_relevance, score)
        # print(true_relevance, score, ranking_loss)
        augment_ranking_losses.append(round(ranking_loss, 4))

    augment_ranking_loss = np.mean(augment_ranking_losses)
    print(augment_ranking_loss)
    metrics_df["augment_ranking_loss"] = augment_ranking_losses

    augment_true_relevance = np.asarray([[0, 0, 1]])
    augment_map_scores = []
    for score in augment_scores:
        score = np.asarray([score])
        map_score = label_ranking_average_precision_score(augment_true_relevance, score)
        # print(true_relevance, score, map_score)
        augment_map_scores.append(round(map_score, 4))

    augment_map_score = np.mean(augment_map_scores)
    print(augment_map_score)
    metrics_df["augment_map_score"] = augment_map_scores

    return metrics_df

metrics_df = run_metrics(score_df)
# metrics_df.to_csv("scores/zwj-metrics.csv")
# metrics_df.to_excel("scores/zwj-metrics.xlsx")

# 0.6663030303030303
# 0.21212121212121213
# 0.7878787878787878