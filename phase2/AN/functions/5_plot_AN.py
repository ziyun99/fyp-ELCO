import pandas as pd
import json
import numpy as np
import os
import plotly.express as px

from data_filepath_sublevel import data_folder, SCORE_DATA_FILEPATH_JSON

def score_json_to_csv(score_data_filepath_json, score_data_filepath_excel=None):
    """
    Input: Raw data file with scores for all annotations for each concept.
    Output: A dataframe with mean similarity scores for each concetps.

    Use mean to aggregate the scores of all annotations for each concept.
    Save the dataframe to xlsx file.
    """
    f = open(score_data_filepath_json)
    data_dict = json.load(f)

    scores = []
    scores_ranked = []
    pos_scores_avg = []
    semineg_scores_avg = []
    attribute = []
    ratings_stats = [[], [], [], []]
    for concept in data_dict:
        scores.append(data_dict[concept]["scores"])
        scores_ranked.append(data_dict[concept]["scores_ranked"])

        semineg_scores = data_dict[concept]["semineg_score"]
        semineg_scores = np.array(semineg_scores)
        avg = np.mean(semineg_scores, axis=1).round(4)
        semineg_scores_avg.append(avg)

        pos_scores = data_dict[concept]["emoji_annotations_score"]
        pos_scores = np.array(pos_scores)
        avg = np.mean(pos_scores, axis=1).round(4)
        pos_scores_avg.append(avg)

        attribute.append(data_dict[concept]["attribute"])

        ratings_stat = data_dict[concept]["ratings_stats"]
        for i in range(4):
            ratings_stats[i].append(ratings_stat[i])

    concepts = list(data_dict.keys())

    adj = [c.split(" ")[0] for c in concepts]

    scores = np.array(scores)
    semineg_scores_avg = np.array(semineg_scores_avg)
    pos_scores_avg = np.array(pos_scores_avg)

    augment_id = 0
    randneg = scores[:, augment_id, 0]
    semineg = scores[:, augment_id, 1]
    semineg_avg = semineg_scores_avg[:, augment_id]
    baseline = scores[:, augment_id, 2]
    pos = scores[:, augment_id, 3]
    pos_avg = pos_scores_avg[:, augment_id]

    augment_id = 1
    augment_randneg = scores[:, augment_id, 0]
    augment_semineg = scores[:, augment_id, 1]
    augment_semineg_avg = semineg_scores_avg[:, augment_id]
    augment_baseline = scores[:, augment_id, 2]
    augment_pos = scores[:, augment_id, 3]
    augment_pos_avg = pos_scores_avg[:, augment_id]

    score_df = pd.DataFrame()
    score_df["concept"] = concepts
    score_df["adj"] = adj
    score_df["attribute"] = attribute

    score_df["randneg"] = randneg
    score_df["semineg_max"] = semineg
    score_df["semineg_avg"] = semineg_avg
    score_df["baseline"] = baseline
    score_df["pos_max"] = pos
    score_df["pos_avg"] = pos_avg

    score_df["augment_randneg"] = augment_randneg
    score_df["augment_semineg_max"] = augment_semineg
    score_df["augment_semineg_avg"] = augment_semineg_avg
    score_df["augment_baseline"] = augment_baseline
    score_df["augment_pos_max"] = augment_pos
    score_df["augment_pos_avg"] = augment_pos_avg
    score_df["ratings_stat_mean"] = ratings_stats[0]
    score_df["ratings_stat_median"] = ratings_stats[1]
    score_df["ratings_stat_mode"] = ratings_stats[2]
    score_df["ratings_stat_variance"] = ratings_stats[3]

    # print(score_df.head())
    if score_data_filepath_excel:
        score_df.to_excel(score_data_filepath_excel)
        print(
            "\nDone reading raw data file: {}, \n returns a mean similarity score dataframe, \n also saved at {}".format(
                score_data_filepath_json, score_data_filepath_excel
            )
        )
    return score_df


def plot_scores(filepath, dataset_name, col, img_name):
    score_df = score_json_to_csv(filepath)
    columns = score_df.columns.values.tolist()
    columns.remove("concept")
    for col in columns:
        img_path = "figures/{}/{}_{}.png".format(dataset_name, dataset_name, col)
        fig = px.bar(score_df, y=[col])
        fig.write_image(img_path, format="png", width=600, height=350, scale=2)


def plot_line_graph(filepath, dataset_name, col, img_name, img_folder):
    score_df = score_json_to_csv(filepath)
    img_path = os.path.join(
        img_folder,
        "figures",
        "{}_{}.png".format(dataset_name, img_name),
    )
    colors = {"randneg": "#6D78F9", "semineg": "purple", "baseline": "#EF553B", "pos": "#00CC96"}
    color_discrete_sequences = {
        2: [colors["randneg"], colors["baseline"]],
        3: [colors["randneg"], colors["baseline"], colors["pos"]],
        4: [colors["randneg"], colors["semineg"], colors["baseline"], colors["pos"]],
    }
    fig = px.line(
        score_df, y=col, color_discrete_sequence=color_discrete_sequences[len(col)]
    )
    fig.update_layout(
        xaxis_title="AN concepts",
        yaxis_title="similarity scores",
        legend_title="Emoji sequences",
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
    )
    fig.write_image(img_path, format="png", width=600, height=350, scale=2)

    print(" Graph saved to: {}".format(img_path))

def plot_graph(score_data_filepath_json, img_folder):
    print("=> Plot graph from data file: {}".format(score_data_filepath_json))

    dataset_name = "AN"

    col = ["randneg", "baseline", "pos_max"]
    img_name = "line_graph_all"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = ["randneg", "semineg_max", "baseline", "pos_max"]
    img_name = "line_graph_all_semineg"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = ["augment_randneg", "augment_baseline", "augment_pos_max"]
    img_name = "line_graph_all_augment"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = [
        "augment_randneg",
        "augment_semineg_max",
        "augment_baseline",
        "augment_pos_max",
    ]
    img_name = "line_graph_all_augment_semineg"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = ["randneg", "augment_randneg"]
    img_name = "line_graph_randneg"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = ["semineg_max", "augment_semineg_max"]
    img_name = "line_graph_semineg"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = ["baseline", "augment_baseline"]
    img_name = "line_graph_baseline"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

    col = ["pos_max", "augment_pos_max"]
    img_name = "line_graph_pos"
    plot_line_graph(score_data_filepath_json, dataset_name, col, img_name, img_folder)

if __name__ == "__main__":
    ## config ##
    SCORE_DATA_FILEPATH_JSON = "/home/ziyun99/fyp-ELCO/phase2/AN/data/experiment/AN_scoring_model.json"
    IMG_FOLDER = os.path.join(
        data_folder,
        "experiment")
    plot_graph(SCORE_DATA_FILEPATH_JSON, IMG_FOLDER)
