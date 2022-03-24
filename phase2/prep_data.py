import json
from operator import pos
import pandas as pd
import numpy as np
import os

import data_filepath

emoji_joiner = "[EM]"

# each sample is labeled according to {0: pos, 1: baseline, 2: semineg, 3: randneg}
assigned_labels = [0, 1, 2, 3]
assigned_scores = [1.0, 0.5, 0.1, 0.05]
categories = ["emoji_annotations_text", "baseline_text", "semineg_text", "randneg_text"]

data_fields = ["sentence1", "sentence2", "score", "label", "label_name", "emoji_list"]

splitted_names = ["train", "validate", "test"]


def is_emoji_text(text):
    if text[0] == ":" and text[-1] == ":":
        return True
    return False


def format_emoji_text(emoji_texts):
    # convert a list of emojis into a one string by using emoji joiner [EM]
    # for eg.:[[":right_arrow:", "national_park"]] => ["[EM]right arrow[EM]national park[EM]"]

    emoji_texts = [texts for texts in emoji_texts if type(texts) != float]  # remove

    cleaned_emoji_texts = []
    for texts in emoji_texts:
        temp = [
            text
            for text in texts
            if type(text) != float and text != "" and is_emoji_text(text)
        ]
        if len(temp) > 0:
            cleaned_emoji_texts.append(temp)

    formatted_texts = [
        "{1}{0}{1}".format(
            emoji_joiner.join([t[1:-1].replace("_", " ") for t in text]), emoji_joiner
        )
        for text in cleaned_emoji_texts
    ]
    return cleaned_emoji_texts, formatted_texts


def format_an_data(raw_data_filepath, train_datapath_json):
    f = open(raw_data_filepath)
    data_dict = json.load(f)

    an_data = {}
    for field in data_fields:
        an_data[field] = []

    count = 0
    for concept in data_dict:
        concept = " ".join(concept.split())  # remove tab if exists
        # print(count, concept)
        count += 1
        for category, assigned_score, assigned_label in zip(
            categories, assigned_scores, assigned_labels
        ):
            emoji_texts = data_dict[concept][category]
            emoji_texts, emoji_strings = format_emoji_text(emoji_texts)
            for text, string in zip(emoji_texts, emoji_strings):
                an_data["sentence1"].append(concept)
                an_data["sentence2"].append(string)
                an_data["score"].append(assigned_score)
                an_data["label"].append(assigned_label)
                an_data["label_name"].append(category)
                an_data["emoji_list"].append(text)

    assert (
        len(an_data["sentence1"])
        == len(an_data["sentence2"])
        == len(an_data["score"])
        == len(an_data["label"])
        == len(an_data["label_name"])
        == len(an_data["emoji_list"])
    )

    with open(train_datapath_json, "w") as outfile:
        json.dump(an_data, outfile, indent=4, allow_nan=True)

    print(
        "\n=> Format raw data into train data.\n  Input file: {}, Save to file: {}".format(
            raw_data_filepath, train_datapath_json
        )
    )
    print("-> Number of data samples: {}".format(len(an_data["sentence1"])))
    sample_output = []
    for key, val in an_data.items():
        sample_output.append("{}: {}".format(key, val[0]))
    sample_output = "\n  ".join(sample_output)
    print("-> Sample data format: \n  {} \n".format(sample_output))


def train_test_split(train_datapath_json, train_datapath_csv, train_folder):
    an_df = pd.read_json(train_datapath_json)
    an_df.to_csv(train_datapath_csv)  # save to csv (of all data points)

    print(
        "\n=> Input file: {}, Save to: {}".format(
            train_datapath_json, train_datapath_csv
        )
    )

    train_df = pd.DataFrame()
    validate_df = pd.DataFrame()
    test_df = pd.DataFrame()

    print("=> Train, validate, test split. Ratio: 60, 20, 20.")
    print("=> Total nummber of samples {}".format(len(an_df)))
    # train, validate, test split is stratified over classes of samples by their labels.
    # each train, validate, test set has equal ratio of [positive, baseline, semineg, randneg] samples.
    for label in assigned_labels:
        sub_sample_df = an_df.loc[an_df["label"] == label]
        train, validate, test = np.split(
            sub_sample_df.sample(frac=1, random_state=42),
            [int(0.6 * len(sub_sample_df)), int(0.8 * len(sub_sample_df))],
        )
        print(
            "  {} {} => train {}, validate {}, test {}".format(
                categories[label],
                sub_sample_df.shape[0],
                train.shape[0],
                validate.shape[0],
                test.shape[0],
            )
        )

        train_df = train_df.append(train)
        validate_df = validate_df.append(validate)
        test_df = test_df.append(test)

    splitted_df = [train_df, validate_df, test_df]
    for df, df_name in zip(splitted_df, splitted_names):
        save_path = os.path.join(train_folder, "{}.csv".format(df_name))
        df.to_csv(save_path)

    print(
        "-> Total samples: train {}, validate {}, test {}".format(
            train_df.shape[0], validate_df.shape[0], test_df.shape[0]
        )
    )
    print(
        "  Save to: {}/train.csv, {}/validate.csv, {}/test.csv\n".format(
            train_folder, train_folder, train_folder
        )
    )


def parallel_data(train_datapath_csv, train_folder):
    an_df = pd.read_csv(train_datapath_csv)

    ## only build parallel data from positive samples
    pos_samples = an_df.loc[an_df["label_name"] == "emoji_annotations_text"]

    pos_samples["parallel data"] = (
        pos_samples["sentence1"].astype(str)
        + "\t"
        + pos_samples["sentence2"].astype(str)
    )

    parallel_data_all = pos_samples["parallel data"].tolist()

    parallel_datafile = os.path.join(train_folder, "parallel_data.txt")
    with open(parallel_datafile, "w") as outfile:
        outfile.write("\n".join(parallel_data_all))

    print_str = []
    for df_name in splitted_names:
        splitted_data_filepath = os.path.join(train_folder, "{}.csv".format(df_name))
        an_df = pd.read_csv(splitted_data_filepath)
        pos_samples = an_df.loc[
            an_df["label_name"] == "emoji_annotations_text"
        ]  ## only build parallel data from positive samples
        pos_samples["parallel data"] = (
            pos_samples["sentence1"].astype(str)
            + "\t"
            + pos_samples["sentence2"].astype(str)
        )
        parallel_data = pos_samples["parallel data"].tolist()

        parallel_data_filepath = os.path.join(
            train_folder, "parallel_data_{}.txt".format(df_name)
        )
        with open(parallel_data_filepath, "w") as outfile:
            outfile.write("\n".join(parallel_data))
        print_str.append(
            "  {} {}. Save to file: {}".format(
                df_name, len(parallel_data), parallel_data_filepath
            )
        )

    print(
        "\n=> Generate parallel dataset. Only from positive data samples. \n  Input file: {}. Save to: {}/parallel_data.txt".format(
            train_datapath_csv, train_folder
        )
    )
    print("  Total samples {}".format(len(parallel_data_all)))
    print("=> Train, validate, test parallel dataset. \n  Number of samples:")
    print("  \n".join(print_str))


def information_retrieval_data(train_datapath_csv, ir_datapath):
    information_retrieval_data = {}

    df = pd.read_csv(train_datapath_csv)
    pos_df = df.loc[
        (df["label_name"] == "emoji_annotations_text")
        | (df["label_name"] == "baseline_text")
    ]  ## only do IR on positive samples

    queries = list(set(pos_df["sentence1"].tolist()))  ## get unique list of queries
    queries.sort()
    information_retrieval_data["ir_queries"] = {i: q for i, q in enumerate(queries)}
    ir_queries_to_idx = {q: i for i, q in enumerate(queries)}
    corpus = list(set(pos_df["sentence2"].tolist()))  ## get unique list of corpus
    information_retrieval_data["ir_corpus"] = {i: c for i, c in enumerate(corpus)}
    ir_corpus_to_idx = {c: i for i, c in enumerate(corpus)}

    ir_relevant_docs = {}
    for _, row in pos_df.iterrows():
        q = row["sentence1"]
        c = row["sentence2"]
        qid = ir_queries_to_idx[q]
        cid = ir_corpus_to_idx[c]
        if qid not in ir_relevant_docs:
            ir_relevant_docs[qid] = []
        if cid not in ir_relevant_docs[qid]:
            ir_relevant_docs[qid].append(cid)
    information_retrieval_data["ir_relevant_docs"] = ir_relevant_docs

    with open(ir_datapath, "w") as outfile:
        json.dump(information_retrieval_data, outfile, indent=4, allow_nan=True)

    print(
        "\n => Generate Information Retrieval data.\n  Input file: {}. Save to: {}".format(
            train_datapath_csv, ir_datapath
        )
    )
    print(
        "  Number of queries {}, nummber of corpus {}, nummber of relavant docs {}".format(
            len(information_retrieval_data["ir_queries"]),
            len(information_retrieval_data["ir_corpus"]),
            len(information_retrieval_data["ir_relevant_docs"]),
        )
    )


if __name__ == "__main__":
    format_an_data(
        data_filepath.RAW_DATA_FILEPATH_JSON, data_filepath.AN_TRAIN_DATAPATH_JSON
    )
    train_test_split(
        data_filepath.AN_TRAIN_DATAPATH_JSON,
        data_filepath.AN_TRAIN_DATAPATH_CSV,
        data_filepath.AN_TRAIN_FOLDER,
    )
    parallel_data(data_filepath.AN_TRAIN_DATAPATH_CSV, data_filepath.AN_TRAIN_FOLDER)
    information_retrieval_data(
        data_filepath.AN_TRAIN_DATAPATH_CSV, data_filepath.AN_IR_DATAPATH
    )
