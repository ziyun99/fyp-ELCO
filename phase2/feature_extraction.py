import json
import os

import torch
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cpu")

from data_filepath import TRAIN_DATA_FOLDER

DATA_FILEPATH = os.path.join(TRAIN_DATA_FOLDER, "parallel_data.txt")


EM = "[EM]"  # Special emoji token


def parse(emoji_sent):
    """
    Parse a string emoji sentence into a list of valid emoji names.

    Extracts all emoji strings from emoji sentence, then replace whitespaces
    in each emoji string with underscore and ":" prefix & postfix.

    E.g. `parse("[EM]dollar banknote[EM]clapping hands[EM]")`
        -> `[":dollar_banknote:", ":clapping_hands:"]`
    """
    result = []
    emoji_sent = emoji_sent.strip(EM).split(EM)
    for emoji_str in emoji_sent:
        sub_emojis = emoji_str.split(" ")
        emoji = ":" + "_".join(sub_emojis) + ":"
        result.append(emoji)
    return result


def getEmbeddings(emoji_sent, model, EM_ID):
    """
    Encode a string emoji sentence using the loaded model then extract all
    [EM] token's embedding found in the sentence as tensors.
    """
    # Get a list of token ids
    tokenizer = model._first_module().tokenizer
    token_ids = tokenizer.encode(emoji_sent)

    # Find indices of all [EM] tokens
    token_idx = []
    for idx, token_id in enumerate(token_ids):
        if token_id == EM_ID:
            token_idx.append(idx)

    # Encode sentence using the loaded model
    codex = model.encode(
        emoji_sent, output_value="token_embeddings", convert_to_tensor=True
    )
    # Asserts that 1 token corresponds to 1 codex (model output)
    assert len(token_ids) == len(codex)

    # Extract all [EM] codex
    embeddings = []
    for idx in token_idx:
        embeddings.append(codex[idx])
    return torch.stack(embeddings)


def main(MODEL_FOLDER, FEATURES_FILEPATH):
    """
    Load pre-trained model from MODEL_FOLDER.
    """
    model = SentenceTransformer(MODEL_FOLDER, DEVICE)
    print(f"Loaded model: {MODEL_FOLDER}\n")
    # print(model)

    with open(os.path.join(MODEL_FOLDER, "added_tokens.json")) as f:
        token_ids = json.load(f)
        EM_ID = token_ids[EM]

    """
    Load data from DATA_FILEPATH.

    A corpus is a list of (english_sentence, emoji_sentence) string tuples.
    """
    corpus = []
    with open(DATA_FILEPATH) as f:
        for row in f.readlines():
            corpus.append(row.strip().split("\t"))

    print(f"Loaded corpus: {DATA_FILEPATH} with {len(corpus)} samples\n")
    # for i in range(5):
    #     print(f"Sample {i + 1}: {corpus[i]}")

    """
    Extract features from corpus
    Features is a list of size len(corpus). Each item in the list is a 
    dict(
        "english_sent": str,
        "emoji_sent": str,
        "emoji_tokens": list[str],
        "num_emoji": int,
        "embeddings": list[list[int]],
        "mean_embedding": list[int],
    ).
    """
    features = []
    for english_sent, emoji_sent in corpus:
        data = dict()
        data["english_sent"] = english_sent
        data["emoji_sent"] = emoji_sent

        emoji_tokens = parse(emoji_sent)
        data["emoji_tokens"] = emoji_tokens
        data["num_emoji"] = len(emoji_tokens)

        data["embeddings"] = getEmbeddings(emoji_sent, model, EM_ID)
        data["mean_embedding"] = torch.mean(data["embeddings"], dim=0)

        assert len(data["embeddings"]) == data["num_emoji"] + 1, data

        features.append(data)

    print(f"Extracted features from {len(features)} samples\n")
    # for i in range(5):
    #     print(f"Sample {i + 1}: {features[i]}\n")
    # print()

    """
    Save features using pickle
    """
    torch.save(features, FEATURES_FILEPATH)
    print(f"Saved features into file: {FEATURES_FILEPATH}\n")


if __name__ == "__main__":
    # Change line below to use other model/checkpoints
    checkpoint_range = 50
    for i in range(checkpoint_range, checkpoint_range*13, checkpoint_range):
        model_file = "model-bert-xlm/checkpoints/{}".format(i)
        MODEL_FOLDER = os.path.join("output", "multilingual", model_file)
        EXPERIMENT_FOLDER = os.path.join(MODEL_FOLDER, "experiment")
        if not os.path.exists(EXPERIMENT_FOLDER):
            os.makedirs(EXPERIMENT_FOLDER)
        FEATURES_FILEPATH = os.path.join(EXPERIMENT_FOLDER, "extracted_features.pt")

        main(MODEL_FOLDER, FEATURES_FILEPATH)
