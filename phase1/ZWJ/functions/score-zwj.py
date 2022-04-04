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

# load sbert model
model_idx = 2
model_name = "/home/ziyun99/fyp-ELCO/phase2/output/multilingual/model-bert-xlm"
pretrained_model = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", model_name]
model = SentenceTransformer(pretrained_model[model_idx], device='cuda:1') 

# load data dict from json file
f = open("/home/ziyun99/fyp-ELCO/phase1/ZWJ/data/raw/zwj-selected.json")
data_dict = json.load(f)

def format_emoji_text(emoji_texts):
    return [" ".join([text[1:-1].replace('_', ' ') for text in emoji_texts]) ] 

concept_count = 0
for concept in data_dict:
    print(concept_count, concept)
    zwj_split_text = data_dict[concept]["zwj_split_text"]
    baseline_text = data_dict[concept]["baseline_text"]
    randneg_text = data_dict[concept]["randneg_text"]

    definition = data_dict[concept]["definition"]
    defined_concept = concept + ' means ' + definition
    concept_augments = [concept, defined_concept]

    zwj_split_text = format_emoji_text(zwj_split_text)
    baseline_text = format_emoji_text(baseline_text)
    randneg_text = format_emoji_text(randneg_text)
    print(zwj_split_text, baseline_text, randneg_text)
    
    #Encode all sentences
    pos_embeddings = model.encode(zwj_split_text)
    baseline_embeddings = model.encode(baseline_text)
    randneg_embeddings = model.encode(randneg_text)

    data_dict[concept]['zwj_score'] = []
    data_dict[concept]['baseline_score'] = []
    data_dict[concept]['randneg_score'] = []
    data_dict[concept]['scores'] = []
    data_dict[concept]['scores_ranked'] = []

    for encoded_concept in concept_augments:
        anchor_embeddings = model.encode(encoded_concept)

        #Compute cosine similarity between all pairs
        pos_sim = util.cos_sim(anchor_embeddings, pos_embeddings).tolist()[0]
        pos_sim = [round(s, 4) for s in pos_sim][0]
        # print(pos_sim)

        baseline_sim = util.cos_sim(anchor_embeddings, baseline_embeddings).tolist()[0]
        baseline_sim = [round(s, 4) for s in baseline_sim][0]
        # print(baseline_sim)

        randneg_sim = util.cos_sim(anchor_embeddings, randneg_embeddings).tolist()[0]
        randneg_sim = [round(s, 4) for s in randneg_sim][0]
        # print(randneg_sim)

        scores = [randneg_sim, baseline_sim, pos_sim]
        scores_ranked = [sorted(scores).index(x) for x in scores]
        print(scores)
        print(scores_ranked)

        data_dict[concept]['zwj_score'].append(pos_sim)
        data_dict[concept]['baseline_score'].append(baseline_sim)
        data_dict[concept]['randneg_score'].append(randneg_sim)

        data_dict[concept]['scores'].append(scores)
        data_dict[concept]['scores_ranked'].append(scores_ranked)

    concept_count += 1
    # if concept_count > 2:
    #     break

end_time = time.time()
total_time = end_time - start_time
print('concept_count: {}, total_time: {}'.format(concept_count, total_time))  

# # save to json file
with open("/home/ziyun99/fyp-ELCO/phase2/AN/data/experiment/ZWJ-scoring.json", "w") as outfile:
    json.dump(data_dict, outfile, indent = 4, allow_nan = True) 

#         total_count += 1
#         if  randneg_sim > baseline_sim:
#             sbert_rank[i] = -2
#             print("randneg_sample wins: {}, {}".format(concept, randneg_text))
#             randneg_count += 1
#         elif baseline_sim > pos_sim:
#             sbert_rank[i] = 0
#             print("baseline wins: {}, {}".format(concept, baseline_text))
#             baseline_count += 1
#         else: 
#             sbert_rank[i] = 1
#             print("pos_sample wins: {}, {}".format(concept, zwj_split_text))
#             pos_count += 1
        
#         sbert_scores.append([pos_sim, baseline_sim, randneg_sim])

# print(randneg_count, baseline_count, pos_count, total_count)  
# #MiniLM: 2 12 19 33
# #mpnet: 3 13 17 33
