import os

# {0: phase1 data, 1: phase2 data}
DATASET_ID = 1

DATA_FOLDER = [
    "/home/ziyun99/fyp-ELCO/phase1/AN/data",
    "/home/ziyun99/fyp-ELCO/phase2/AN/data",
]

data_folder = DATA_FOLDER[DATASET_ID]

# ground truth attributes of adjectives
ATTRIBUTE_GROUND_TRUTH = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_attribute_ground_truth.csv"
)

# source file to be parsed
RAW_DATA_FILEPATH_EXCEL = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_data_collection.xlsx"
)

# target file to be saved
RAW_DATA_FILEPATH_JSON = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_data_collection.json"
)



# model to get raw similarity scores of all annotations
MODEL_ID = 0

MODEL_NAME = ["mpnet", "MiniLM"]
PRETRAINED_MODEL = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]

model_name = PRETRAINED_MODEL[MODEL_ID]

SCORE_DATA_FILEPATH_JSON = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_scoring_{}.json".format(MODEL_NAME[MODEL_ID])
)


# aggregated scores based on concepts, attributes and adj
SCORE_DATA_FILEPATH_EXCEL = os.path.join(DATA_FOLDER[DATASET_ID], "scores", "AN_score.xlsx")

ATTRIBUTE_DATA_FILEPATH = os.path.join(
    DATA_FOLDER[DATASET_ID], "scores", "AN_score_attribute.xlsx"
)

ADJ_DATA_FILEPATH = os.path.join(DATA_FOLDER[DATASET_ID], "scores", "AN_score_adj.xlsx")


AN_TRAIN_FOLDER = os.path.join(DATA_FOLDER[DATASET_ID], "training")
AN_TRAIN_DATAPATH_JSON = os.path.join(AN_TRAIN_FOLDER, "AN_data.json")
AN_TRAIN_DATAPATH_CSV = os.path.join(AN_TRAIN_FOLDER, "AN_data.csv")

AN_IR_DATAPATH = os.path.join(AN_TRAIN_FOLDER, "information_retrieval_data.json")


TRAIN_DATA_FOLDER = os.path.join(data_folder, "training")