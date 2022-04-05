import json
import logging
import os

import pandas as pd
from data_filepath import data_folder
from sentence_transformers import SentenceTransformer, util

EXPERIMENT_NAMES = ["ir-original", "ir-exp", "ir-exp-shuffled"]
DATA_NAMES = ["training/information_retrieval_data.json", "experiment/ir_data.json", "experiment/shuffled_ir_data.json"]

###### CONFIG ######
EXPERIMENT_ID = 2
EXPERIMENT_NAME = EXPERIMENT_NAMES[EXPERIMENT_ID]
MODEL_NAME = "model-bert-xlm"
DATA_NAME = DATA_NAMES[EXPERIMENT_ID]

###### LOAD MODEL ######
gpu_id = 0
DEVICE = "cuda:{}".format(gpu_id)
MODEL_FOLDER = os.path.join("output", "multilingual", MODEL_NAME)
model = SentenceTransformer(MODEL_FOLDER, DEVICE)

###### LOAD IR DATA ######
logging.info("=> Load information retrieval dataset")
input_file = os.path.join(data_folder, DATA_NAME)

information_retrieval_data = json.load(open(input_file))
ir_queries = information_retrieval_data["ir_queries"]
ir_corpus = information_retrieval_data["ir_corpus"]
ir_relevant_docs = information_retrieval_data["ir_relevant_docs"]

###### LOAD PARALLEL DATA ######
logging.info("=> Load information retrieval dataset")

set_names = ["train", "test", "validate"]
parallel_data = {set_name:[] for set_name in set_names}

for set_name in set_names:
    input_file = os.path.join(data_folder, f"training/parallel_data_{set_name}.txt")
    fp = open(input_file, "r")
    parallel_data[set_name] = list(fp.read().splitlines())
    fp.close()

###### SCORE ######
columns = ["query", "relevant_doc", "relevant_score", "set_name"]
score_dict = {c:list() for c in columns}

for qid, query in ir_queries.items():
    qembed = model.encode(query)
    relevant_doc_ids = ir_relevant_docs[qid]
    for cid in relevant_doc_ids:
        relevant_doc = ir_corpus[str(cid)]
        cembed = model.encode(relevant_doc)
        cos_sim = util.cos_sim(qembed, cembed).item()
        score_dict["query"].append(query)
        score_dict["relevant_doc"].append(relevant_doc)
        score_dict["relevant_score"].append(cos_sim)
        hit = False
        sent = '\t'.join([query, relevant_doc])
        for set_name in set_names:
            if sent in parallel_data[set_name]:
                score_dict["set_name"].append(set_name)
                print(set_name)
                hit = True
                break
        if not hit:
            score_dict["set_name"].append("baseline")

df = pd.DataFrame.from_dict(score_dict)
score_csv_path = os.path.join(f'experiment/{EXPERIMENT_NAME}-score.csv')
df.to_csv(score_csv_path)

