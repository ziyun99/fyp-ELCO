import os


DATASET_ID = 1

DATA_FOLDER = [
    "/home/ziyun99/fyp-ELCO/phase1/AN/data",
    "/home/ziyun99/fyp-ELCO/phase2/AN/data",
]

data_folder = DATA_FOLDER[DATASET_ID]

# source file to be parsed
RAW_DATA_FILEPATH_EXCEL = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_data_collection.xlsx"
)

# target file to be saved
RAW_DATA_FILEPATH_JSON = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_data_collection.json"
)

ATTRIBUTE_GROUND_TRUTH = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_attribute_ground_truth.csv"
)


SCORE_DATA_FILEPATH_EXCEL = os.path.join(DATA_FOLDER[DATASET_ID], "scores", "AN_score.xlsx")

ATTRIBUTE_DATA_FILEPATH = os.path.join(
    DATA_FOLDER[DATASET_ID], "scores", "AN_score_attribute.xlsx"
)

ADJ_DATA_FILEPATH = os.path.join(DATA_FOLDER[DATASET_ID], "scores", "AN_score_adj.xlsx")


MODEL_ID = 0

MODEL_NAME = ["mpnet", "MiniLM"]
PRETRAINED_MODEL = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]

model_name = PRETRAINED_MODEL[MODEL_ID]

SCORE_DATA_FILEPATH_JSON = os.path.join(
    DATA_FOLDER[DATASET_ID], "raw", "AN_scoring_{}.json".format(MODEL_NAME[MODEL_ID])
)
