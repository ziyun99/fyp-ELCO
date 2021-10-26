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
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1')


# load dict from json file
f = open("data.json")
data_dict = json.load(f)

concept = ""
emoji_annotations_text = []
neg_sample_text = ""

pos_count = 0
total_count = 0

for concept in data_dict:
    print(total_count, concept)
    emoji_annotations_text = data_dict[concept]["emoji_annotations_text"]
    neg_sample_text = data_dict[concept]["neg_sample_text"]
    rating_annotations = data_dict[concept]["rating_annotations"]
    # print(emoji_annotations_text, neg_sample_text)

    emoji_annotations_text = [" ".join([t.replace('_', ' ') for t in text]) for text in emoji_annotations_text if type(text) != float] 
    neg_sample_text = [" ".join([t.replace('_', ' ') for t in text]) for text in neg_sample_text if type(text) != float]
    
    mean = st.mean(rating_annotations)
    median = st.median(rating_annotations)
    mode = st.mode(rating_annotations)
    variance = st.variance(rating_annotations, mean)
    ratings_stats = [mean, median, mode, variance]
    ratings_stats = [round(r, 4) for r in ratings_stats] 
    data_dict[concept]['ratings_stats'] = ratings_stats
    # print("mean, median, mode, variance", mean, median, mode, variance)  
    print(emoji_annotations_text, neg_sample_text)


    #Encode all sentences
    base_embeddings = model.encode(concept)
    pos_embeddings = model.encode(emoji_annotations_text)
    neg_embeddings = model.encode(neg_sample_text)

    #Compute cosine similarity between all pairs
    pos_sim = util.cos_sim(base_embeddings, pos_embeddings).tolist()[0]
    pos_sim = [round(s, 4) for s in pos_sim]
    max_pos = max(pos_sim)
    max_pos_idx = pos_sim.index(max_pos)
    # print(pos_sim)

    neg_cos_sim = util.cos_sim(base_embeddings, neg_embeddings).tolist()[0]
    neg_cos_sim = [round(s, 4) for s in neg_cos_sim]
    max_neg = max(neg_cos_sim)
    max_neg_idx = neg_cos_sim.index(max_neg)
    # print(neg_cos_sim)

    if max_pos > max_neg:
        data_dict[concept]['sbert_rank'] = 1
        print("pos sample wins: ", concept, emoji_annotations_text[max_pos_idx])
        pos_count += 1
    else: 
        data_dict[concept]['sbert_rank'] = 0
        print("neg sample wins: ", concept, neg_sample_text[max_neg_idx])
    total_count += 1

    data_dict[concept]['emoji_annotations_score'] = pos_sim
    data_dict[concept]['neg_sample_score'] = neg_cos_sim

    # if total_count > 16:
    #     break

print(pos_count, total_count)  # 162 210

end_time = time.time()
total_time = end_time - start_time
print('total_time:',  total_time)

# # converto dict to json
# json_object = json.dumps(data_dict[concept], indent = 4, allow_nan = True) 
# print(json_object)

# save to json file
with open("data_scoring.json", "w") as outfile:
    json.dump(data_dict, outfile, indent = 4, allow_nan = True) 
