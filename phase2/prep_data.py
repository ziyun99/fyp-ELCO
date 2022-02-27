import json
from operator import pos
import pandas as pd
import numpy as np

emoji_joiner = "[EM]"

def format_emoji_text(emoji_texts):
    emoji_texts = [text for text in emoji_texts if type(text) != float] 
    return emoji_texts, [ '{1}{0}{1}'.format(emoji_joiner.join([t[1:-1].replace('_', ' ') for t in text]), emoji_joiner) for text in emoji_texts if type(text) != float] 


categories = ["emoji_annotations_text", "baseline_text", "semineg_text", "randneg_text"]
assigned_scores = [1.0, 0.5, 0.1, 0.05]
assigned_labels = [0, 1, 2, 3]

data_fields = ['sentence1', 'sentence2', 'score', 'label', 'label_name', 'emoji_list']

an_data_filepath = "../data/AN/an-data-collection.json"
an_formatted_data_filepath = "../data/training/an_data.json"

def format_an_data():
    f = open(an_data_filepath)
    data_dict = json.load(f)

    an_data = {}
    for field in data_fields:
        an_data[field] = []

    count = 0
    for concept in data_dict:
        concept = " ".join(concept.split())  # remove tab if exists
        print(count, concept)
        count += 1
        for category, assigned_score, assigned_label in zip(categories, assigned_scores, assigned_labels):
            emoji_texts = data_dict[concept][category]
            emoji_texts, emoji_strings = format_emoji_text(emoji_texts)
            for text, string in zip(emoji_texts, emoji_strings):
                an_data['sentence1'].append(concept)
                an_data['sentence2'].append(string)
                an_data['score'].append(assigned_score)
                an_data['label'].append(assigned_label)
                an_data['label_name'].append(category)
                an_data['emoji_list'].append(text)

    with open(an_formatted_data_filepath, "w") as outfile:
        json.dump(an_data, outfile, indent = 4, allow_nan = True) 

splitted_names = ["train", "validate", "test"]

def train_test_split():
    an_df = pd.read_json(an_formatted_data_filepath)
    an_df.to_csv("../data/training/an_data.csv")  # save to csv (of all data points)

    train_df = pd.DataFrame()
    validate_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for label in assigned_labels:
        print(label)
        sub_sample_df = an_df.loc[an_df['label'] == label]
        train, validate, test = np.split(sub_sample_df.sample(frac=1, random_state=42), [int(.6*len(sub_sample_df)), int(.8*len(sub_sample_df))])
        print(sub_sample_df.shape)
        print(train.shape, validate.shape, test.shape)
        
        train_df = train_df.append(train)
        validate_df = validate_df.append(validate)
        test_df = test_df.append(test)
        
    print(train_df.shape, validate_df.shape, test_df.shape)

    splitted_df = [train_df, validate_df, test_df]
    for df, df_name in zip(splitted_df, splitted_names):
        filepath = "../data/training/{}.csv".format(df_name)
        df.to_csv(filepath)

def parallel_data():
    for df_name in splitted_names:
        an_df = pd.read_csv("../data/training/{}.csv".format(df_name))
        pos_samples = an_df.loc[an_df['label_name'] == "emoji_annotations_text"] ## only build parallel data from positive samples
        pos_samples["parallel data"] = pos_samples["sentence1"].astype(str) + '\t' + pos_samples["sentence2"].astype(str)
        parallel_data = pos_samples["parallel data"].tolist()

        parallel_data_filepath = "../data/training/parallel_data_{}.txt".format(df_name)
        with open(parallel_data_filepath, "w") as outfile:
            outfile.write("\n".join(parallel_data))

def information_retrieval_data():
    information_retrieval_data = {}

    df = pd.read_csv("../data/training/an_data.csv")
    pos_df = df.loc[(df['label_name'] == "emoji_annotations_text") | (df['label_name'] == "baseline_text")]  ## only do IR on positive samples

    queries = list(set(pos_df['sentence1'].tolist())) ## get unique list of queries
    queries.sort()
    information_retrieval_data["ir_queries"] = {i: q for i, q in enumerate(queries)}
    ir_queries_to_idx = {q: i for i, q in enumerate(queries)}
    corpus = list(set(pos_df['sentence2'].tolist())) ## get unique list of corpus
    information_retrieval_data["ir_corpus"] = {i: c for i, c in enumerate(corpus)}
    ir_corpus_to_idx = {c: i for i, c in enumerate(corpus)}

    ir_relevant_docs = {}
    for _, row in pos_df.iterrows():
        q = row['sentence1']
        c = row['sentence2']
        qid = ir_queries_to_idx[q]
        cid = ir_corpus_to_idx[c]
        if qid not in ir_relevant_docs:
            ir_relevant_docs[qid] = []
        if cid not in ir_relevant_docs[qid]:
            ir_relevant_docs[qid].append(cid)
    information_retrieval_data["ir_relevant_docs"] = ir_relevant_docs

    with open("../data/training/information_retrieval_data.json", "w") as outfile:
        json.dump(information_retrieval_data, outfile, indent = 4, allow_nan = True) 
        
if __name__ == "__main__":
    format_an_data()
    train_test_split()
    # parallel_data()
    # information_retrieval_data()

# train_test_split
# 0
# (1153, 6)
# (691, 6) (231, 6) (231, 6)
# 1
# (210, 6)
# (126, 6) (42, 6) (42, 6)
# 2
# (1574, 6)
# (944, 6) (315, 6) (315, 6)
# 3
# (210, 6)
# (126, 6) (42, 6) (42, 6)
# (1887, 6) (630, 6) (630, 6)