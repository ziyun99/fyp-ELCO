import json
import os
import logging

from sentence_transformers import SentenceTransformer, util
from evaluator import ir_eval

from data_filepath import data_folder, TRAIN_DATA_FOLDER

EXPERIMENT_NAMES = ["ir-original", "ir-exp-k5", "ir-exp-shuffled-k5"]
DATA_NAMES = ["training/information_retrieval_data.json", "experiment/ir_data.json", "experiment/shuffled_ir_data.json"]

###### CONFIG ######
EXPERIMENT_ID = 1
EXPERIMENT_NAME = EXPERIMENT_NAMES[EXPERIMENT_ID]
MODEL_NAME = "model-bert-xlm"
DATA_NAME = DATA_NAMES[EXPERIMENT_ID]


###### MAIN FILEPATH ######
EXPERIMENT_FOLDER = os.path.join("experiment", EXPERIMENT_NAME)
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)

###### SET LOGGER ######
LOG_FILEPATH = os.path.join(EXPERIMENT_FOLDER, "logfile")
logging.basicConfig(
    filename=LOG_FILEPATH, filemode="w", format="%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# log the config
logging.info(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
logging.info(f"MODEL_NAME: {MODEL_NAME}")
logging.info(f"DATA_NAME: {DATA_NAME}")


###### LOAD MODEL ######
gpu_id = 0
DEVICE = "cuda:{}".format(gpu_id)
MODEL_FOLDER = os.path.join("output", "multilingual", MODEL_NAME)
model = SentenceTransformer(MODEL_FOLDER, DEVICE)

###### LOAD DATA ######
logging.info("=> Load information retrieval dataset")
input_file = os.path.join(data_folder, DATA_NAME)

information_retrieval_data = json.load(open(input_file))
ir_queries = information_retrieval_data["ir_queries"]
ir_corpus = information_retrieval_data["ir_corpus"]
ir_relevant_docs = information_retrieval_data["ir_relevant_docs"]

# log the size of data
logging.info(
    "  Input file: {}.\n  Number of queries {}, nummber of corpus {}, nummber of relavant docs {}".format(
        input_file,
        len(information_retrieval_data["ir_queries"]),
        len(information_retrieval_data["ir_corpus"]),
        len(information_retrieval_data["ir_relevant_docs"]),
    )
)

# convert list of relevant docs to set
for key, value in ir_relevant_docs.items():
    ir_relevant_docs[key] = set([str(v) for v in value])

information_retrieval_evaluator = ir_eval.InformationRetrievalEvaluator(
    ir_queries, ir_corpus, ir_relevant_docs, name="inforet-test", score_functions={'cos_sim': util.cos_sim}
)

###### RUN EVALUATOR ######
information_retrieval_evaluator(model, EXPERIMENT_FOLDER)
