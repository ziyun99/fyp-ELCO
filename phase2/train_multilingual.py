import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, evaluation, losses, models
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from data_filepath import DATASET_ID, TRAIN_DATA_FOLDER

output_path = "output/multilingual/model-" + datetime.now().strftime("%Y-%m-%d_%H-%M")

gpu_id = 0
device = "cuda:{}".format(gpu_id)

#### Just some code to print debug information to stdout
log_filename = output_path + "/logfile"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
# handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
logging.info("=> Input train data from folder: {}".format(TRAIN_DATA_FOLDER))

###### CREATE MODEL ######
teacher_model = ["bert-base-nli-stsb-mean-tokens", "all-mpnet-base-v2"]
teacher_model_name = teacher_model[0]  # Our monolingual teacher model, we want to convert to multiple languages
student_model_name = (
    "xlm-roberta-base"  # Multilingual base model we use to imitate the teacher model
)

max_seq_length = 128  # Student model max. lengths for inputs (number of word pieces)
train_batch_size = 32  # Batch size for training
inference_batch_size = 32  # Batch size at inference

num_epochs = 600  # Train for x epochs
num_warmup_steps = 5000  # Warumup steps

num_evaluation_steps = 1000  # Evaluate performance after every xxxx steps

num_epochs_to_save = 50  # Save the model checkpoint after xxxx epochs

# Load teacher model
logging.info("=> Load teacher model")
teacher_model = SentenceTransformer(teacher_model_name, device=device)

# Create student model
logging.info("=> Create student model")
word_embedding_model = models.Transformer(student_model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], device=device
)


###### Add special token for emoji ######
tokens = ["[EM]"]

word_embedding_model = model._first_module()
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(
    len(word_embedding_model.tokenizer)
)


###### Load parallel dataset: train, validate, test sets ######
logging.info("=> Load parallel dataset")

splitted_names = ["train", "validate", "test"]
parallel_data_filepaths = {}
for df_name in splitted_names:
    input_file = os.path.join(TRAIN_DATA_FOLDER, "parallel_data_{}.txt".format(df_name))
    parallel_data_filepaths[df_name] = input_file
    fp = open(input_file, "r")
    number_of_lines = len(fp.readlines())
    fp.close()
    logging.info("{}: {}. input file: {}".format(df_name, number_of_lines, input_file))

train_reader = ParallelSentencesDataset(
    student_model=model, teacher_model=teacher_model
)
train_reader.load_data(parallel_data_filepaths["train"])
train_dataloader_mse = DataLoader(
    train_reader, shuffle=True, batch_size=train_batch_size
)
train_loss_mse = losses.MSELoss(model=model)


evaluators = []
for df_name in ["train", "validate", "test"]:
    src_sentences = []
    trg_sentences = []
    with open(
        os.path.join(TRAIN_DATA_FOLDER, "parallel_data_{}.txt".format(df_name)),
        "rt",
        encoding="utf8",
    ) as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            if splits[0] != "" and splits[1] != "":
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    mse_evaluator = evaluation.MSEEvaluator(
        src_sentences,
        trg_sentences,
        name="mse-" + df_name,
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(mse_evaluator)

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
    trans_acc_evaluator = evaluation.TranslationEvaluator(
        src_sentences,
        trg_sentences,
        name="trans-" + df_name,
        batch_size=inference_batch_size,
    )
    evaluators.append(trans_acc_evaluator)


###### Load cosine similarity loss dataset: train, validate, test sets ######
logging.info("=> Load cosine similarity dataset")
sim_samples = {}
for df_name in splitted_names:
    input_file = os.path.join(TRAIN_DATA_FOLDER, "{}.csv".format(df_name))
    df = pd.read_csv(input_file)
    sentence1 = df["sentence1"].tolist()
    sentence2 = df["sentence2"].tolist()
    score = df["score"].tolist()

    sim_samples[df_name] = [
        InputExample(texts=[s1, s2], label=s)
        for s1, s2, s in zip(sentence1, sentence2, score)
    ]
    logging.info("{}: {}. input file: {}".format(df_name, len(df), input_file))

train_dataloader_sim = DataLoader(
    sim_samples["train"], shuffle=True, batch_size=train_batch_size
)
train_loss_sim = losses.CosineSimilarityLoss(model=model)

sim_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    sim_samples["train"], name="cossim-train"
)
evaluators.append(sim_evaluator)
sim_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    sim_samples["validate"], name="cossim-validate"
)
evaluators.append(sim_evaluator)
sim_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    sim_samples["test"], name="cossim-test"
)
evaluators.append(sim_evaluator)

###### Load constrastive loss dataset: train, validate, test sets ######
# print("Load constrastive dataset")
# constrastive_samples = {}
# for df_name in splitted_names:
#     df = pd.read_csv("../data/training/{}.csv".format(df_name))

#     pos_df = df.loc[df['label'] == "emoji_annotations_text"]
#     sentence1 = pos_df['sentence1'].tolist()
#     sentence2 = pos_df['sentence2'].tolist()

#     constrastive_samples[df_name] = [InputExample(texts=[s1, s2], label=1) for s1,s2 in zip(sentence1, sentence2)]

#     neg_df = df.loc[df['label'] != "emoji_annotations_text"]
#     sentence1 = neg_df['sentence1'].tolist()
#     sentence2 = neg_df['sentence2'].tolist()
#     constrastive_samples[df_name] += [InputExample(texts=[s1, s2], label=0) for s1,s2 in zip(sentence1, sentence2)]

# train_dataloader_contrastive = DataLoader(constrastive_samples["train"], shuffle=True, batch_size=train_batch_size)
# train_loss_contrastive = losses.ContrastiveLoss(model=model)

# contrastive_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(constrastive_samples["validate"], name='constrastive-val')
# evaluators.append(contrastive_evaluator)
# contrastive_evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(constrastive_samples["test"], name='constrastive-test')
# evaluators.append(contrastive_evaluator)


# ###### Load triplet loss dataset: train, validate, test sets ######
# print("Load triplet dataset")


###### Load information retrieval (IR) loss dataset: test set ######
logging.info("=> Load information retrieval dataset")
input_file = os.path.join(TRAIN_DATA_FOLDER, "information_retrieval_data.json")

information_retrieval_data = json.load(open(input_file))
ir_queries = information_retrieval_data["ir_queries"]
ir_corpus = information_retrieval_data["ir_corpus"]
ir_relevant_docs = information_retrieval_data["ir_relevant_docs"]

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

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# metrices. For our use case MRR@k and Accuracy@k are relevant.
information_retrieval_evaluator = evaluation.InformationRetrievalEvaluator(
    ir_queries, ir_corpus, ir_relevant_docs, name="inforet-test"
)  # , accuracy_at_k=[1220], precision_recall_at_k=[1220])
evaluators.append(information_retrieval_evaluator)

logging.info("=> Evaluators:")
for i, e in enumerate(evaluators):
    logging.info("{}: {}".format(i, e.name))

###### Train model ######
train_objectives = [
    (train_dataloader_mse, train_loss_mse),
    (train_dataloader_sim, train_loss_sim),
]  # , (train_dataloader_contrastive, train_loss_contrastive)]

steps_per_epoch = {0: 28, 1: 37}

model.fit(
    train_objectives=train_objectives,
    evaluator=evaluation.SequentialEvaluator(
        evaluators,
        main_score_function=lambda scores: np.mean([scores[2], scores[3], scores[7]]),
    ),  # np.mean(scores)) [scores[0], scores[1], scores[4], scores[6]])),
    epochs=num_epochs,
    evaluation_steps=num_evaluation_steps,
    warmup_steps=num_warmup_steps,
    scheduler="warmuplinear",
    output_path=output_path,
    save_best_model=True,
    optimizer_params={"lr": 2e-5, "eps": 1e-6, "correct_bias": False},
    checkpoint_save_steps=steps_per_epoch[DATASET_ID] * num_epochs_to_save,
    checkpoint_path=output_path + "/checkpoints",
)
