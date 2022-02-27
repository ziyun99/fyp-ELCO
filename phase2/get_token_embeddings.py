from sentence_transformers.datasets import ParallelSentencesDataset
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, losses, evaluation, readers, util
from sentence_transformers.readers import InputExample

from sentence_transformers import LoggingHandler, SentenceTransformer, models
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import logging, os

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


###### CREATE MODEL ######

# Load base model
print("Load base model")
base_model_name = 'xlm-roberta-base'       #Multilingual base model we use to imitate the teacher model
word_embedding_model = models.Transformer(base_model_name)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda:1')
###### Add special token for emoji to base model ######
tokens = ["[EM]"]
word_embedding_model = base_model._first_module()
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

# Load trained model
model_path = "output/multilingual/model-2022-02-17_13-40"
print("Load model")
model = SentenceTransformer(model_path, device='cuda:1')
print(model)


###### Test tokenizer ######
# EM_token = '[EM]'
# tokenizer = model._first_module().tokenizer
# print(len(tokenizer)) 

# sentences = ['[EM]money[ENG] [EMO]wings[EM][EM]worker[EM]][QRY]][DOC] increasing[EM]cityscape[EM][KIK]😄🐶']
# decoded_input = tokenizer.decode(tokenizer.encode(sentences[0]))
# print(decoded_input)
# encoded_input = tokenizer(sentences[0])#, padding=True, truncation=True, return_tensors='pt')
# print(encoded_input)
# decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
# print(decoded_tokens)


# EM_indices = [i for i, x in enumerate(decoded_tokens) if x == EM_token]
# EM_idx = [encoded_input.input_ids[i] for i in EM_indices]
# print(EM_indices)
# print(EM_idx)

# EM_indices = [i for i, x in enumerate(encoded_input.input_ids) if x == 250002]
# EM_idx = [decoded_tokens[i] for i in EM_indices]
# print(EM_indices)
# print(EM_idx)



embedders = [base_model, model]
# Corpus with example sentences
corpus = ["[EM]dollar banknote[EM]dollar banknote[EM]Japanese open for business button[EM]",
          "[EM]dollar banknote[EM]briefcase[EM]",
          "[EM]black large square[EM]cloud[EM]",
          "[EM]repeat button[EM]repeat single button[EM]",
          "[EM]money with wings[EM]man office worker[EM]chart increasing[EM]cityscape[EM]",
          "[EM]grinning face[EM]man[EM]",
          "[EM]person in suit levitating light skin tone[EM]",
          "[EM]radio[EM]",
          'busy man',
          'idle man',
          'rich man',
          "poor man",
          "big party"
          ]
# Query sentences:
queries = ['big business']
top_k = len(corpus)

for embedder in embedders:
    embedder.eval()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))


# # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
# hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
# hits = hits[0]      #Get the hits for the first query
# for hit in hits:
#     print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


###### Load information retrieval (IR) loss dataset: test set ######
# print("Load information retrieval dataset")
# information_retrieval_data = json.load(open("../data/training/information_retrieval_data.json"))
# ir_queries = information_retrieval_data["ir_queries"]
# ir_corpus = information_retrieval_data["ir_corpus"]
# ir_relevant_docs = information_retrieval_data["ir_relevant_docs"]

# # convert list of relevant docs to set
# for key, value in ir_relevant_docs.items():
#     ir_relevant_docs[key] = set(value)

# # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# # metrices. For our use case MRR@k and Accuracy@k are relevant.
# information_retrieval_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs, name='inforet-test', accuracy_at_k=[1220], precision_recall_at_k=[1220])
# res = model.evaluate(information_retrieval_evaluator)
# print(res)