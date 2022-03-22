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
dataset_idx = 1 
dataset_name = datasets[dataset_idx]

def get_scores_ranked():
    f = open("data/AN/an-scoring-mpnet.json")
    data_dict = json.load(f)

    rank = []
    rank_semineg = []
    for concept in data_dict:
        scores = [[s[0]] + s[2:] for s in data_dict[concept]["scores"]]
        scores_ranked = [[sorted(score).index(x) for x in score] for score in scores]
        scores_ranked_semineg = data_dict[concept]["scores_ranked"]
        for i in scores_ranked:
            if 2 not in i:
                print(scores_ranked)
                break
        rank.append(scores_ranked)

        scores_semineg = data_dict[concept]["scores"]
        scores_ranked_semineg = data_dict[concept]["scores_ranked"]
        rank_semineg.append(scores_ranked_semineg)
        # print(scores, scores_ranked, scores_semineg, scores_ranked_semineg)
        
    return rank, rank_semineg

def rank_eval():
    ranks, rank_semineg = get_scores_ranked()
    # print(ranks)
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

    ranks = rank_semineg
    for i in [0, 1]:
        print(i)
        randneg = [1 for rank in ranks if rank[i][0] == 3]
        print("randneg", len(randneg))

        semineg = [1 for rank in ranks if rank[i][1] == 3]
        print("semineg", len(semineg))

        baseline = [1 for rank in ranks if rank[i][2] == 3]
        print("baseline", len(baseline))

        pos = [1 for rank in ranks if rank[i][3] == 3]
        print("pos", len(pos))


        randneg_semineg = [1 for rank in ranks if rank[i][0] > rank[i][1]]
        print("randneg > semineg", len(randneg_semineg))

        randneg_baseline = [1 for rank in ranks if rank[i][0] > rank[i][2]]
        print("randneg > baseline", len(randneg_baseline))    

        randneg_pos = [1 for rank in ranks if rank[i][0] > rank[i][3]]
        print("randneg > pos", len(randneg_pos))

        semineg_baseline = [1 for rank in ranks if rank[i][1] > rank[i][2]]
        print("semineg > baseline", len(semineg_baseline))    

        semineg_pos = [1 for rank in ranks if rank[i][1] > rank[i][3]]
        print("semineg > pos", len(semineg_pos))

        baseline_pos = [1 for rank in ranks if rank[i][2] > rank[i][3]]
        print("baseline > pos", len(baseline_pos))


# df = pd.read_csv("scores/AN-score.csv")
# df = df.drop(columns=["semineg_avg", "augment_semineg_avg", "semineg_max", "augment_semineg_max", "pos_avg", "augment_pos_avg"])
# print(df.shape)


# # randneg
# # semineg_max
# # baseline
# # pos_max

# df2 = df.loc[(df['baseline'] > df['pos_max']) & (df['augment_pos_max'] > df['augment_baseline'])]
# df1 = df.loc[(df['pos_max'] > df['baseline']) & (df['augment_baseline'] > df['augment_pos_max'])]
# print(df1)
# print(df2)
# print(len(df1), len(df2))

# s1 = pd.merge(df1, df2, how='inner')
# print(s1)

# set_diff_df = pd.concat([df2, df1, df1]).drop_duplicates(keep=False)
# print(set_diff_df)

target_col = 'adj'
target='bright'

df = pd.read_excel("scores/AN-score.xlsx")
print(df.shape)
win = df.loc[(df[target_col] == target) & (df['randneg'] > df['semineg_max']) & (df['randneg'] > df['baseline']) & (df['randneg'] > df['pos_max'])]
print(len(win))

win = df.loc[(df[target_col] == target) & (df['semineg_max'] > df['randneg']) & (df['semineg_max'] > df['baseline']) & (df['semineg_max'] > df['pos_max'])]
print(len(win))

win = df.loc[(df[target_col] == target) & (df['baseline'] > df['randneg']) & (df['baseline'] > df['semineg_max']) & (df['baseline'] > df['pos_max'])]
print(len(win))

win = df.loc[(df[target_col] == target) & (df['pos_max'] > df['randneg']) & (df['pos_max'] > df['semineg_max']) & (df['pos_max'] > df['baseline'])]
print(len(win))
# augment_randneg
# augment_semineg_max
# augment_baseline
# augment_pos_max

win = df.loc[(df[target_col] == target) & (df['augment_randneg'] > df['augment_semineg_max']) & (df['augment_randneg'] > df['augment_baseline']) & (df['augment_randneg'] > df['augment_pos_max'])]
print(len(win))

win = df.loc[(df[target_col] == target) & (df['augment_semineg_max'] > df['augment_randneg']) & (df['augment_semineg_max'] > df['augment_baseline']) & (df['augment_semineg_max'] > df['augment_pos_max'])]
print(len(win))

win = df.loc[(df[target_col] == target) & (df['augment_baseline'] > df['augment_randneg']) & (df['augment_baseline'] > df['augment_semineg_max']) & (df['augment_baseline'] > df['augment_pos_max'])]
print(len(win))

win = df.loc[(df[target_col] == target) & (df['augment_pos_max'] > df['augment_randneg']) & (df['augment_pos_max'] > df['augment_semineg_max']) & (df['augment_pos_max'] > df['augment_baseline'])]
print(len(win))

target_col = 'adj'
target='domestic'
win = df.loc[(df[target_col] == target)]
print(win)

# Attributes
# total 77
# randneg 0
# baseline 10
# pos_max 67

# ATTRIBUTE-AUGMENT
# total 77
# randneg 0
# baseline 10
# pos_max 67

# Adj
# total 45
# randneg 0
# baseline 3
# pos_max 42

# ADJ-AUGMENT
# total 45
# randneg 0
# baseline 3
# pos_max 42