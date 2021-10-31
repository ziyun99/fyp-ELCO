
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
baseline = scores[:, augment_id, 2]
pos = scores[:, augment_id, 3]

semineg_avg = semineg_scores_avg[:, augment_id]
pos_avg = pos_scores_avg[:, augment_id]


augment_id = 1
augment_randneg = scores[:, augment_id, 0]
augment_semineg = scores[:, augment_id, 1]
augment_baseline = scores[:, augment_id, 2]
augment_pos = scores[:, augment_id, 3]

augment_semineg_avg = semineg_scores_avg[:, augment_id]
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
score_df.to_csv("scores/AN-score.csv")

columns = score_df.columns.values.tolist()
columns.remove('concept')

import plotly.express as px
for col in columns:
    img_path = "figures/AN/{}.png".format(col)
    fig = px.bar(score_df,  y=[col])
    fig.write_image(img_path, format="png", width=600, height=350, scale=2)