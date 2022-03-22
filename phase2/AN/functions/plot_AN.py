import pandas as pd
import json
import numpy as np

filepath = "../data/AN_scoring_mpnet.json"
old_path = "/home/ziyun99/fyp-ELCO/data/AN/AN_scoring_mpnet.json"


def score_json_to_csv(filepath):
    f = open(filepath)
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
    concept_count = len(concepts)

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
    return score_df


score_df = score_json_to_csv(filepath)
score_df.to_csv("../scores/AN_score.csv")
score_df.to_excel("../scores/AN_score.xlsx")


def compute_matching_baseline(filepath):
    f = open(filepath)
    data_dict = json.load(f)

    all_match_counts = []
    num_response = 0
    emoji_len = 0
    non_zeros = 0

    perfect_match_count = 0
    for concept in data_dict:
        emoji_annotations_text = data_dict[concept]["emoji_annotations_text"]
        baseline_text = data_dict[concept]["baseline_text"][0]
        match_counts = [
            len(set(em) & set(baseline_text))
            for em in emoji_annotations_text
            if type(em) != float
        ]
        if 2 in match_counts:
            # print(emoji_annotations_text, baseline_text)
            perfect_match_count += 1
        avg_match_counts = np.mean(np.array(match_counts))
        all_match_counts.append(avg_match_counts)

        response = len(match_counts)
        num_response += response

        emoji_len_avg = np.mean(
            np.array([len(em) for em in emoji_annotations_text if type(em) != float])
        )
        emoji_len += emoji_len_avg

        non_zeros += np.count_nonzero(match_counts)

    avg_all_match_counts = np.mean(all_match_counts)
    zero_emoji_match = num_response - non_zeros
    one_emoji_match = non_zeros - perfect_match_count

    print("=> Outcome on the file: {}\n".format(filepath))
    print(
        "total number of responses from annotators: {}, average nummber of annotaters per concept: {:.2f}".format(
            num_response, num_response / len(data_dict)
        )
    )
    print("avg length of emoji annotations: {:.2f}".format(emoji_len / len(data_dict)))
    print("avg match count: {:.2f}".format(avg_all_match_counts))
    print(
        "number of non-zero match count (for all the emoji annotations): {}".format(
            non_zeros
        )
    )
    print(
        "number of zero-emoji match count: {} ({:.2f}%)".format(
            zero_emoji_match, zero_emoji_match / num_response * 100
        )
    )
    print(
        "number of one-emoji match count: {} ({:.2f}%)".format(
            one_emoji_match, one_emoji_match / num_response * 100
        )
    )
    print(
        "number of two-emoji match count: {} ({:.2f}%)".format(
            perfect_match_count, perfect_match_count / num_response * 100
        )
    )


# compute_matching_baseline(filepath)


def compute_ratings(filepath):
    f = open(filepath)
    data_dict = json.load(f)

    rating_stats = []
    for concept in data_dict:
        rating_stats.append(data_dict[concept]["ratings_stats"])

    rating_stats = np.array(rating_stats)
    rating_stats_mean = np.nanmean(rating_stats, axis=0)

    print("=> Outcome on the file: {}\n".format(filepath))
    print(
        "Ratings (over all annotations of all concepts): mean, median, mode, variance",
        rating_stats_mean,
    )


# compute_ratings(old_path)


# generate & save csv files on mean scores & margin scores 
# which are aggregated by attribute and adjective of the AN concepts
# and sort the concepts by margin score
def groupby_adj_attri(filepath):
    score_df = score_json_to_csv(filepath)
    print(score_df)
    score_df = score_df.drop(
        columns=["semineg_avg", "augment_semineg_avg", "pos_avg", "augment_pos_avg"]
    )

    # compute margin scores
    score_df["pos_baseline"] = score_df["pos_max"] - score_df["baseline"]
    score_df["pos_semineg"] = score_df["pos_max"] - score_df["semineg_max"]
    score_df["pos_randneg"] = score_df["pos_max"] - score_df["randneg"]

    # group by attributes
    count = score_df.groupby("attribute")["pos_max"].transform("count")
    score_df["attri_count"] = count

    mean_bygroup = score_df.groupby("attribute").mean().round(4)
    mean_bygroup = mean_bygroup.sort_values(
        by=["pos_baseline", "pos_randneg", "pos_semineg", "attri_count"],
        ascending=False,
    )
    print(mean_bygroup)
    mean_bygroup.to_excel("../scores/breakdown/AN-score-attri.xlsx")

    # group by adjectives
    count = score_df.groupby("adj")["pos_max"].transform("count")
    score_df["adj_count"] = count

    mean_bygroup = score_df.groupby("adj").mean().round(4)
    mean_bygroup = mean_bygroup.sort_values(
        by=["pos_baseline", "pos_randneg", "pos_semineg", "adj_count"], ascending=False
    )
    print(mean_bygroup.shape)
    mean_bygroup.to_excel("../scores/breakdown/AN-score-adj.xlsx")


groupby_adj_attri(filepath)


def get_top10_ratings(filepath):
    df = score_json_to_csv(filepath)
    target_col = "attribute"
    target = [
        "PERFECTION",
        "FRESHNESS",
        "WEIGHT",
        "DOMESTICITY",
        "DULLNESS",
        "LEGALITY",
        "COMPLETENESS",
        "RIGHTNESS",
        "COLOR",
        "ORDINARINESS",
    ]
    target_col = "adj"
    target = [
        "fresh",
        "foreign",
        "heavy",
        "internal",
        "domestic",
        "sound",
        "intelligent",
        "far",
        "full",
        "bright",
    ]
    for t in target:
        win = df.loc[(df[target_col] == t)]["ratings_stats"].to_numpy()
        win = [i[0] for i in win]
        win = np.mean(np.array(win))
        print(round(win, 4))


import plotly.express as px


def plot_scores(score_df):
    columns = score_df.columns.values.tolist()
    columns.remove("concept")
    for col in columns:
        img_path = "figures/{}/{}_{}.png".format(dataset_name, dataset_name, col)
        fig = px.bar(score_df, y=[col])
        fig.write_image(img_path, format="png", width=600, height=350, scale=2)


def plot_line_graph(score_df, col, img_name):
    img_path = "figures/{}/{}_{}.png".format(dataset_name, dataset_name, img_name)
    fig = px.line(
        score_df, y=col, color_discrete_sequence=["blue", "purple", "red", "green"]
    )
    fig.update_layout(
        xaxis_title="AN concepts",
        yaxis_title="similarity scores",
        legend_title="Emoji sequences",
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
    )
    fig.write_image(img_path, format="png", width=600, height=350, scale=2)


# score_df = read_score_df()
# col = ['randneg', 'semineg_max', 'baseline','pos_max']
# img_name = "line_graph_semineg_all"
# plot_line_graph(score_df, col, img_name)
# plot_scores(score_df)


from sklearn.metrics import ndcg_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score


def run_metrics(score_df):
    metrics_df = score_df
    randneg = [np.array(score_df["randneg"])]
    semineg = [np.array(score_df["semineg_max"])]
    semineg_avg = [np.array(score_df["semineg_avg"])]
    baseline = [np.array(score_df["baseline"])]
    pos = [np.array(score_df["pos_max"])]
    pos_avg = [np.array(score_df["pos_avg"])]

    scores = np.concatenate((randneg, semineg, baseline, pos), axis=0).T
    print(scores.shape)

    true_relevance = np.asarray([[-1, -0.5, 0, 1]])
    ndcg_scores = []
    for score in scores:
        score = np.asarray([score])
        ndcg = ndcg_score(true_relevance, score)
        # print(true_relevance, score, ndcg)
        ndcg_scores.append(round(ndcg, 4))

    ndcg = np.mean(ndcg_scores)
    print(ndcg)
    metrics_df["ndcg_score"] = ndcg_scores

    true_relevance = np.asarray([[0, 0, 0, 1]])
    ranking_losses = []
    for score in scores:
        score = np.asarray([score])
        ranking_loss = label_ranking_loss(true_relevance, score)
        # print(true_relevance, score, ranking_loss)
        ranking_losses.append(round(ranking_loss, 4))

    ranking_loss = np.mean(ranking_losses)
    print(ranking_loss)
    metrics_df["ranking_loss"] = ranking_losses

    true_relevance = np.asarray([[0, 0, 0, 1]])
    map_scores = []
    for score in scores:
        score = np.asarray([score])
        map_score = label_ranking_average_precision_score(true_relevance, score)
        # print(true_relevance, score, map_score)
        map_scores.append(round(map_score, 4))

    map_score = np.mean(map_scores)
    print(map_score)
    metrics_df["map_score"] = map_scores

    augment_randneg = [np.array(score_df["augment_randneg"])]
    augment_semineg = [np.array(score_df["augment_semineg_max"])]
    augment_semineg_avg = [np.array(score_df["augment_semineg_avg"])]
    augment_baseline = [np.array(score_df["augment_baseline"])]
    augment_pos = [np.array(score_df["augment_pos_max"])]
    augment_pos_avg = [np.array(score_df["augment_pos_avg"])]

    augment_scores = np.concatenate(
        (augment_randneg, augment_semineg, augment_baseline, augment_pos), axis=0
    ).T
    print(augment_scores.shape)

    augment_true_relevance = np.asarray([[-1, -0.5, 0, 1]])
    augment_ndcg_scores = []
    for score in augment_scores:
        score = np.asarray([score])
        ndcg = ndcg_score(augment_true_relevance, score)
        # print(true_relevance, score, ndcg)
        augment_ndcg_scores.append(round(ndcg, 4))

    augment_ndcg = np.mean(augment_ndcg_scores)
    print(augment_ndcg)
    metrics_df["augment_ndcg_score"] = augment_ndcg_scores

    augment_true_relevance = np.asarray([[0, 0, 0, 1]])
    augment_ranking_losses = []
    for score in augment_scores:
        score = np.asarray([score])
        ranking_loss = label_ranking_loss(augment_true_relevance, score)
        # print(augment_true_relevance, score, ranking_loss)
        augment_ranking_losses.append(round(ranking_loss, 4))

    augment_ranking_loss = np.mean(augment_ranking_losses)
    print(augment_ranking_loss)
    metrics_df["augment_ranking_loss"] = augment_ranking_losses

    augment_true_relevance = np.asarray([[0, 0, 0, 1]])
    augment_map_scores = []
    for score in augment_scores:
        score = np.asarray([score])
        map_score = label_ranking_average_precision_score(augment_true_relevance, score)
        # print(true_relevance, score, map_score)
        augment_map_scores.append(round(map_score, 4))

    augment_map_score = np.mean(augment_map_scores)
    print(augment_map_score)
    metrics_df["augment_map_score"] = augment_map_scores

    return metrics_df


# metrics_df = run_metrics(score_df)
# metrics_df.to_csv("scores/AN-metrics.csv")
# metrics_df.to_excel("scores/AN-metrics.xlsx")

# 0.3946842857142857
# 0.11904285714285713
# 0.84722
