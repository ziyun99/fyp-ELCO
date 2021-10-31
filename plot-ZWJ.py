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
score_df.to_csv("scores/zwj-score.csv")

columns = score_df.columns.values.tolist()
columns.remove('concept')

import plotly.express as px
for col in columns:
    img_path = "figures/ZWJ/{}.png".format(col)
    fig = px.bar(score_df,  y=[col])
    fig.write_image(img_path, format="png", width=600, height=350, scale=2)

