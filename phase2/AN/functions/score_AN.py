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
model_idx = 1
pretrained_model = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
model = SentenceTransformer(pretrained_model[model_idx], device='cuda:1') 

# load data dict from json file
f = open("AN_data_collection.json")
data_dict = json.load(f)

def format_emoji_text(emoji_texts):
    return [" ".join([t[1:-1].replace('_', ' ') for t in text]) for text in emoji_texts if type(text) != float] 

concept_count = 0
for concept in data_dict:
    print(concept_count, concept)
    emoji_annotations_text = data_dict[concept]["emoji_annotations_text"]
    baseline_text = data_dict[concept]["baseline_text"]
    semineg_text = data_dict[concept]["semineg_text"]
    randneg_text = data_dict[concept]["randneg_text"]
    rating_annotations = data_dict[concept]["rating_annotations"]
    # print(emoji_annotations_text, baseline_text)

    attribute = data_dict[concept]["attribute"].lower()
    attributed_concept = concept + ' is about ' + attribute
    # definition = data_dict[concept]["definition"]
    # defined_concept = concept + ' means ' + definition
    concept_augments = [concept, attributed_concept] #, defined_concept]

    emoji_annotations_text = format_emoji_text(emoji_annotations_text)
    baseline_text = format_emoji_text(baseline_text)
    semineg_text = format_emoji_text(semineg_text)
    randneg_text = format_emoji_text(randneg_text)
    # print(emoji_annotations_text, baseline_text, semineg_text, randneg_text)
    
    mean = st.mean(rating_annotations)
    median = st.median(rating_annotations)
    mode = st.mode(rating_annotations)
    variance = st.variance(rating_annotations, mean)
    ratings_stats = [mean, median, mode, variance]
    ratings_stats = [round(r, 4) for r in ratings_stats] 
    data_dict[concept]['ratings_stats'] = ratings_stats
    # print("mean, median, mode, variance", mean, median, mode, variance)  

    #Encode all sentences
    pos_embeddings = model.encode(emoji_annotations_text)
    baseline_embeddings = model.encode(baseline_text)
    semineg_embeddings = model.encode(semineg_text)
    randneg_embeddings = model.encode(randneg_text)

    data_dict[concept]['emoji_annotations_score'] = []
    data_dict[concept]['baseline_score'] = []
    data_dict[concept]['semineg_score'] = []
    data_dict[concept]['randneg_score'] = []
    data_dict[concept]['scores'] = []
    data_dict[concept]['scores_ranked'] = []

    for encoded_concept in concept_augments:
        anchor_embeddings = model.encode(encoded_concept)

        #Compute cosine similarity between all pairs
        pos_sim = util.cos_sim(anchor_embeddings, pos_embeddings).tolist()[0]
        pos_sim = [round(s, 4) for s in pos_sim]
        max_pos = max(pos_sim)
        max_pos_idx = pos_sim.index(max_pos)
        # print(pos_sim)

        baseline_sim = util.cos_sim(anchor_embeddings, baseline_embeddings).tolist()[0]
        baseline_sim = [round(s, 4) for s in baseline_sim]
        max_baseline = max(baseline_sim)
        max_baseline_idx = baseline_sim.index(max_baseline)
        # print(baseline_sim)

        semineg_sim = util.cos_sim(anchor_embeddings, semineg_embeddings).tolist()[0]
        semineg_sim = [round(s, 4) for s in semineg_sim]
        max_semineg = max(semineg_sim)
        max_semineg_idx = semineg_sim.index(max_semineg)
        # print(semineg_sim)

        randneg_sim = util.cos_sim(anchor_embeddings, randneg_embeddings).tolist()[0]
        randneg_sim = [round(s, 4) for s in randneg_sim]
        max_randneg = max(randneg_sim)
        max_randneg_idx = randneg_sim.index(max_randneg)
        # print(randneg_sim)

        scores = [max_randneg, max_semineg, max_baseline, max_pos]
        scores_ranked = [sorted(scores).index(x) for x in scores]
        print(scores)
        print(scores_ranked)

        data_dict[concept]['emoji_annotations_score'].append(pos_sim)
        data_dict[concept]['baseline_score'].append(baseline_sim)
        data_dict[concept]['semineg_score'].append(semineg_sim)
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
model_name = ["mpnet", "MiniLM"]
filename = "AN-scoring-" + model_name[model_idx] + ".json"
with open(filename, "w") as outfile:
    json.dump(data_dict, outfile, indent = 4, allow_nan = True) 


# print(randneg_count, semineg_count, baseline_count, pos_count, total_count)  
# MiniLM: 162(pos) 210(total) / 114(semineg) 67(pos) 210(total) / 
# mpnet: 172 210 / 115 65 210 / 21 104 27 58 210

# rand, baseline, pos
# MiniLM: 13 44 153 210 vs  2 12 19 33

# augment - MiniLM
# rand ^, semineg , baseline ^, pos
# (before) 12 111 27 60 /210 vs (after) 19 108 38 45 /210
# rand ^, baseline ^, pos
# (before) 13 44 153 /210  vs (after) 19 49 142 /210
