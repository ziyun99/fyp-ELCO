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

def get_scores_ranked():
    f = open("data/ZWJ/zwj-scoring-mpnet.json")
    data_dict = json.load(f)

    scores = []
    scores_ranked = []
    for concept in data_dict:
        scores.append(data_dict[concept]["scores"])
        scores_ranked.append(data_dict[concept]["scores_ranked"])
    return scores_ranked

ranks = get_scores_ranked()
print(ranks)
from collections import Counter

for i in [0, 1]:
    print(i)
    randneg = [1 for rank in ranks if rank[i][0] == 2]
    print("randneg", len(randneg))

    baseline = [1 for rank in ranks if rank[i][1] == 2]
    print("baseline", len(baseline))

    pos = [1 for rank in ranks if rank[i][2] == 2]
    print("pos", len(pos))

    randneg_baseline = [1 for rank in ranks if rank[i][0] > rank[i][1]]
    print("randneg > baseline", len(randneg_baseline))

    randneg_pos = [1 for rank in ranks if rank[i][0] > rank[i][2]]
    print("randneg > pos", len(randneg_pos))

    baseline_pos = [1 for rank in ranks if rank[i][1] > rank[i][2]]
    print("baseline > pos", len(baseline_pos))
