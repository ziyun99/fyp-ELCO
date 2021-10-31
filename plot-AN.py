
from os import read
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

def read_score_df():
    f = open("data/AN/an-scoring-mpnet.json")
    data_dict = json.load(f)

    scores = []
    scores_ranked = []
    pos_scores_avg = []
    semineg_scores_avg = []
    for concept in data_dict:
        scores.append(data_dict[concept]["scores"])
        scores_ranked.append(data_dict[concept]['scores_ranked'])

        semineg_scores = data_dict[concept]['semineg_score']
        semineg_scores = np.array(semineg_scores)
        avg = np.mean(semineg_scores, axis=1).round(4)
        semineg_scores_avg.append(avg)

        pos_scores = data_dict[concept]['emoji_annotations_score']
        pos_scores = np.array(pos_scores)
        avg = np.mean(pos_scores, axis=1).round(4)
        pos_scores_avg.append(avg)

    concepts = list(data_dict.keys())
    concept_count = len(concepts)

    scores = np.array(scores)
    semineg_scores_avg = np.array(semineg_scores_avg)
    pos_scores_avg = np.array(pos_scores_avg)

    augment_id = 0
    randneg = scores[:, augment_id, 0]
    semineg = scores[:, augment_id, 1]
    semineg_avg = semineg_scores_avg[:, augment_id]
    baseline = scores[:, augment_id, 2]
    pos = scores[:, augment_id, 3]
    pos_avg = pos_scores_avg[:, augment_id]


    augment_id = 1
    augment_randneg = scores[:, augment_id, 0]
    augment_semineg = scores[:, augment_id, 1]
    augment_semineg_avg = semineg_scores_avg[:, augment_id]
    augment_baseline = scores[:, augment_id, 2]
    augment_pos = scores[:, augment_id, 3]
    augment_pos_avg = pos_scores_avg[:, augment_id]

    score_df = pd.DataFrame()
    score_df['concept'] = concepts

    score_df['randneg'] = randneg
    score_df['semineg_max'] = semineg
    score_df['semineg_avg'] = semineg_avg
    score_df['baseline'] = baseline
    score_df['pos_max'] = pos
    score_df['pos_avg'] = pos_avg

    score_df['augment_randneg'] = augment_randneg
    score_df['augment_semineg_max'] = augment_semineg
    score_df['augment_semineg_avg'] = augment_semineg_avg
    score_df['augment_baseline'] = augment_baseline
    score_df['augment_pos_max'] = augment_pos
    score_df['augment_pos_avg'] = augment_pos_avg
    
    print(score_df.head())
    return score_df

def plot_scores(score_df):
    columns = score_df.columns.values.tolist()
    columns.remove('concept')

    import plotly.express as px
    for col in columns:
        img_path = "figures/AN/{}.png".format(col)
        fig = px.bar(score_df,  y=[col])
        fig.write_image(img_path, format="png", width=600, height=350, scale=2)

score_df = read_score_df()
# plot_scores(score_df)
# score_df.to_csv("scores/AN-score.csv")

from sklearn.metrics import ndcg_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score

def run_metrics(score_df):
    metrics_df = score_df
    randneg = [np.array(score_df['randneg'])]
    semineg = [np.array(score_df['semineg_max'])]
    semineg_avg = [np.array(score_df['semineg_avg'])]
    baseline = [np.array(score_df['baseline'])]
    pos = [np.array(score_df['pos_max'])]
    pos_avg = [np.array(score_df['pos_avg'])]

    scores = np.concatenate((randneg, semineg, baseline, pos), axis=0).T
    print(scores.shape)


    true_relevance = np.asarray([[-1, -0.5, 0, 1]])
    ndcg_scores = []
    for score in scores:
        score = np.asarray([score])
        ndcg = ndcg_score(true_relevance, score)
        # print(true_relevance, score, ndcg)
        ndcg_scores.append(round(ndcg, 4))

    ndcg = np.mean(ndcg_scores)
    print(ndcg)
    metrics_df["ndcg_score"] = ndcg_scores
    
    true_relevance = np.asarray([[0, 0, 0, 1]])
    ranking_losses = []
    for score in scores:
        score = np.asarray([score])
        ranking_loss = label_ranking_loss(true_relevance, score)
        # print(true_relevance, score, ranking_loss)
        ranking_losses.append(round(ranking_loss, 4))

    ranking_loss = np.mean(ranking_losses)
    print(ranking_loss)
    metrics_df["ranking_loss"] = ranking_losses

    true_relevance = np.asarray([[0, 0, 0, 1]])
    map_scores = []
    for score in scores:
        score = np.asarray([score])
        map_score = label_ranking_average_precision_score(true_relevance, score)
        # print(true_relevance, score, map_score)
        map_scores.append(round(map_score, 4))

    map_score = np.mean(map_scores)
    print(map_score)
    metrics_df["map_score"] = map_scores

    return metrics_df

metrics_df = run_metrics(score_df)
metrics_df.to_csv("scores/AN-metrics.csv")

# 0.3946842857142857
# 0.11904285714285713
# 0.84722