import pandas as pd
import re
import json 
import emoji
import functools
import operator
import statistics as st
import time
from sentence_transformers import SentenceTransformer, util

start_time = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1') #all-mpnet-base-v2  #all-MiniLM-L6-v2


def format_data(list_of_strings):
    list_of_texts = []
    for text in list_of_strings:
        if type(text) != type('str'):
            list_of_texts.append('')
            continue
        text = text[1:-1].split(', ')
        text = [t[1:-1] for t  in text]
        txt = " ".join([t[1:-1].replace('_', ' ') for t in text])
        list_of_texts.append(txt)
    return list_of_texts

# load df from csv file
zwj_df = pd.read_csv("data/zwj-selected.csv", index_col=0)
print(zwj_df.head())
col_names = zwj_df.columns.values
print(col_names)

zwj_concepts_list = zwj_df['name'].tolist()
zwj_split_text_list = zwj_df['zwj_split_text'].tolist()
baseline_text_list = zwj_df['baseline_text'].tolist()
randneg_text_list = zwj_df['randneg_text'].tolist()

zwj_concepts_list = [" ".join([text[1:-1].replace('_', ' ')]) for text in zwj_concepts_list if type(text) != float] 
zwj_split_text_list = format_data(zwj_split_text_list)
baseline_text_list = format_data(baseline_text_list)
randneg_text_list = format_data(randneg_text_list)

# print(zwj_concepts_list)
# print(zwj_split_text_list)
# print(baseline_text_list)
# print(randneg_text_list)

randneg_count = 0
baseline_count = 0
pos_count = 0
total_count = 0

num_concepts = len(zwj_concepts_list)
 
sbert_rank = [2 for i in range(num_concepts)]
sbert_scores = []

for i in range(num_concepts):
    concept = zwj_concepts_list[i]
    zwj_split_text = zwj_split_text_list[i]
    baseline_text = baseline_text_list[i]
    randneg_text = randneg_text_list[i]

    #Encode all sentences
    anchor_embeddings = model.encode(concept)
    pos_embeddings = model.encode(zwj_split_text)
    baseline_embeddings = model.encode(baseline_text)
    randneg_embeddings = model.encode(randneg_text)

    #Compute cosine similarity between all pairs
    pos_sim = util.cos_sim(anchor_embeddings, pos_embeddings).tolist()[0]
    pos_sim = [round(s, 4) for s in pos_sim][0]

    baseline_sim = util.cos_sim(anchor_embeddings, baseline_embeddings).tolist()[0]
    baseline_sim = [round(s, 4) for s in baseline_sim][0]

    randneg_sim = util.cos_sim(anchor_embeddings, randneg_embeddings).tolist()[0]
    randneg_sim = [round(s, 4) for s in randneg_sim][0]

    total_count += 1
    if  randneg_sim > baseline_sim:
        sbert_rank[i] = -2
        print("randneg_sample wins: {}, {}".format(concept, randneg_text))
        randneg_count += 1
    elif baseline_sim > pos_sim:
        sbert_rank[i] = 0
        print("baseline wins: {}, {}".format(concept, baseline_text))
        baseline_count += 1
    else: 
        sbert_rank[i] = 1
        print("pos_sample wins: {}, {}".format(concept, zwj_split_text))
        pos_count += 1
    
    sbert_scores.append([pos_sim, baseline_sim, randneg_sim])

print(randneg_count, baseline_count, pos_count, total_count)  
#MiniLM: 2 12 19 33
#mpnet: 3 13 17 33

end_time = time.time()
total_time = end_time - start_time
print('total_time:',  total_time)

# save scoring to csv file
zwj_df["sbert_scores"] = sbert_scores
zwj_df["sbert_rank"] = sbert_rank
zwj_df.to_csv("data/zwj-scoring.csv")
