import pandas as pd
import json
import numpy as np
import os

from data_filepath import SCORE_DATA_FILEPATH_JSON, SCORE_DATA_FILEPATH_EXCEL, ATTRIBUTE_DATA_FILEPATH, ADJ_DATA_FILEPATH


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


def groupby_adj_attri(score_data_filepath_json, attribute_data_filepath, adj_data_filepath):
    """
    Analysis on scores by breaking down on attribute and adjective.

    Compute mean similarity scores & margin scores,
    which are obtain from the aggregation of attribute and adjective of the AN concepts,
    sort the concepts by margin score,
    and save into xlsx files.
    """

    df = score_json_to_csv(score_data_filepath_json)
    score_df = df.drop(
        columns=["semineg_avg", "augment_semineg_avg", "pos_avg", "augment_pos_avg"]
    )

    # compute margin scores
    score_df["pos_baseline"] = score_df["pos_max"] - score_df["baseline"]
    score_df["pos_semineg"] = score_df["pos_max"] - score_df["semineg_max"]
    score_df["pos_randneg"] = score_df["pos_max"] - score_df["randneg"]

    # group by attributes
    count = score_df.groupby("attribute")["pos_max"].transform("count")
    score_df["attribute_count"] = count
    mean_bygroup = score_df.groupby("attribute").mean().round(4)
    mean_bygroup = mean_bygroup.sort_values(
        by=["pos_baseline", "pos_randneg", "pos_semineg", "attribute_count"],
        ascending=False,
    )
    print(mean_bygroup.shape)
    mean_bygroup.to_excel(attribute_data_filepath)

    # group by adjectives
    count = score_df.groupby("adj")["pos_max"].transform("count")
    score_df["adj_count"] = count
    mean_bygroup = score_df.groupby("adj").mean().round(4)
    mean_bygroup = mean_bygroup.sort_values(
        by=["pos_baseline", "pos_randneg", "pos_semineg", "adj_count"], ascending=False
    )
    print(mean_bygroup.shape)
    mean_bygroup.to_excel(adj_data_filepath)

    print(
        "\nDone reading datafile: {}, \n saved breakdown aggregated scores for \n attributes (total attribute count: {}): {}, \n for adjectives (total adj count: {}): {}".format(
            score_data_filepath_json,
            len(attribute_data_filepath),
            attribute_data_filepath,
            len(adj_data_filepath),
            adj_data_filepath,
        )
    )


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

    print("\n=> Compute matching baseline. Outcome on the file: {}".format(filepath))
    print(
        " total number of responses from annotators: {}, average nummber of annotaters per concept: {:.2f}".format(
            num_response, num_response / len(data_dict)
        )
    )
    print(" avg length of emoji annotations: {:.2f}".format(emoji_len / len(data_dict)))
    print(" avg match count: {:.2f}".format(avg_all_match_counts))
    print(
        " number of non-zero match count (for all the emoji annotations): {}".format(
            non_zeros
        )
    )
    print(
        " number of zero-emoji match count: {} ({:.2f}%)".format(
            zero_emoji_match, zero_emoji_match / num_response * 100
        )
    )
    print(
        " number of one-emoji match count: {} ({:.2f}%)".format(
            one_emoji_match, one_emoji_match / num_response * 100
        )
    )
    print(
        " number of two-emoji match count: {} ({:.2f}%)".format(
            perfect_match_count, perfect_match_count / num_response * 100
        )
    )


def compute_ratings(filepath):
    f = open(filepath)
    data_dict = json.load(f)

    rating_stats = []
    for concept in data_dict:
        rating_stats.append(data_dict[concept]["ratings_stats"])

    rating_stats = np.array(rating_stats)
    rating_stats_mean = np.nanmean(rating_stats, axis=0)

    print("\n=> Compute ratings. Outcome on the file: {}".format(filepath))
    print(
        "Ratings (over all annotations of all concepts): mean, median, mode, variance",
        rating_stats_mean,
    )


def get_top_bottom_k(filepath, k, min_samples, breakdown_type):
    df = pd.read_excel(filepath)
    df = df.loc[df["{}_count".format(breakdown_type)] >= min_samples]
    topk = df[:k]
    bottomk = df[-k:]

    print("\n => Get TopK and Bottom K. Outcome on the file: {}".format(filepath))
    print(
        " Top{} with min {} samples: \n List: {}, margin scores (mean: {:.4f}): {}, average ratings (mean: {:.4f}): {}".format(
            k,
            min_samples,
            topk["{}".format(breakdown_type)].tolist(),
            np.mean(topk["pos_baseline"].tolist()),
            topk["pos_baseline"].tolist(),
            np.mean(topk["ratings_stat_mean"].tolist()),
            topk["ratings_stat_mean"].tolist(),
        )
    )
    print(
        " Bottom{} with min {} samples: \n List: {}, margin scores (mean: {:.4f}): {}, average ratings (mean: {:.4f}): {}".format(
            k,
            min_samples,
            bottomk["{}".format(breakdown_type)].tolist(),
            np.mean(bottomk["pos_baseline"].tolist()),
            bottomk["pos_baseline"].tolist(),
            np.mean(bottomk["ratings_stat_mean"].tolist()),
            bottomk["ratings_stat_mean"].tolist(),
        )
    )

def analyse_scores():
    # aggregate similarity scores by concept
    score_df = score_json_to_csv(SCORE_DATA_FILEPATH_JSON, SCORE_DATA_FILEPATH_EXCEL)

    # aggregae similarity scores by adjectives/attributes
    groupby_adj_attri(SCORE_DATA_FILEPATH_JSON, ATTRIBUTE_DATA_FILEPATH, ADJ_DATA_FILEPATH)

    compute_matching_baseline(SCORE_DATA_FILEPATH_JSON)
    compute_ratings(SCORE_DATA_FILEPATH_JSON)

    get_top_bottom_k(ADJ_DATA_FILEPATH, k=5, min_samples=3, breakdown_type="adj")
    get_top_bottom_k(
        ATTRIBUTE_DATA_FILEPATH, k=5, min_samples=2, breakdown_type="attribute"
    )


if __name__ == "__main__":
    analyse_scores()


# from sklearn.metrics import ndcg_score
# from sklearn.metrics import label_ranking_loss
# from sklearn.metrics import label_ranking_average_precision_score


# def run_metrics(score_df):
#     metrics_df = score_df
#     randneg = [np.array(score_df["randneg"])]
#     semineg = [np.array(score_df["semineg_max"])]
#     semineg_avg = [np.array(score_df["semineg_avg"])]
#     baseline = [np.array(score_df["baseline"])]
#     pos = [np.array(score_df["pos_max"])]
#     pos_avg = [np.array(score_df["pos_avg"])]

#     scores = np.concatenate((randneg, semineg, baseline, pos), axis=0).T
#     print(scores.shape)

#     true_relevance = np.asarray([[-1, -0.5, 0, 1]])
#     ndcg_scores = []
#     for score in scores:
#         score = np.asarray([score])
#         ndcg = ndcg_score(true_relevance, score)
#         # print(true_relevance, score, ndcg)
#         ndcg_scores.append(round(ndcg, 4))

#     ndcg = np.mean(ndcg_scores)
#     print(ndcg)
#     metrics_df["ndcg_score"] = ndcg_scores

#     true_relevance = np.asarray([[0, 0, 0, 1]])
#     ranking_losses = []
#     for score in scores:
#         score = np.asarray([score])
#         ranking_loss = label_ranking_loss(true_relevance, score)
#         # print(true_relevance, score, ranking_loss)
#         ranking_losses.append(round(ranking_loss, 4))

#     ranking_loss = np.mean(ranking_losses)
#     print(ranking_loss)
#     metrics_df["ranking_loss"] = ranking_losses

#     true_relevance = np.asarray([[0, 0, 0, 1]])
#     map_scores = []
#     for score in scores:
#         score = np.asarray([score])
#         map_score = label_ranking_average_precision_score(true_relevance, score)
#         # print(true_relevance, score, map_score)
#         map_scores.append(round(map_score, 4))

#     map_score = np.mean(map_scores)
#     print(map_score)
#     metrics_df["map_score"] = map_scores

#     augment_randneg = [np.array(score_df["augment_randneg"])]
#     augment_semineg = [np.array(score_df["augment_semineg_max"])]
#     augment_semineg_avg = [np.array(score_df["augment_semineg_avg"])]
#     augment_baseline = [np.array(score_df["augment_baseline"])]
#     augment_pos = [np.array(score_df["augment_pos_max"])]
#     augment_pos_avg = [np.array(score_df["augment_pos_avg"])]

#     augment_scores = np.concatenate(
#         (augment_randneg, augment_semineg, augment_baseline, augment_pos), axis=0
#     ).T
#     print(augment_scores.shape)

#     augment_true_relevance = np.asarray([[-1, -0.5, 0, 1]])
#     augment_ndcg_scores = []
#     for score in augment_scores:
#         score = np.asarray([score])
#         ndcg = ndcg_score(augment_true_relevance, score)
#         # print(true_relevance, score, ndcg)
#         augment_ndcg_scores.append(round(ndcg, 4))

#     augment_ndcg = np.mean(augment_ndcg_scores)
#     print(augment_ndcg)
#     metrics_df["augment_ndcg_score"] = augment_ndcg_scores

#     augment_true_relevance = np.asarray([[0, 0, 0, 1]])
#     augment_ranking_losses = []
#     for score in augment_scores:
#         score = np.asarray([score])
#         ranking_loss = label_ranking_loss(augment_true_relevance, score)
#         # print(augment_true_relevance, score, ranking_loss)
#         augment_ranking_losses.append(round(ranking_loss, 4))

#     augment_ranking_loss = np.mean(augment_ranking_losses)
#     print(augment_ranking_loss)
#     metrics_df["augment_ranking_loss"] = augment_ranking_losses

#     augment_true_relevance = np.asarray([[0, 0, 0, 1]])
#     augment_map_scores = []
#     for score in augment_scores:
#         score = np.asarray([score])
#         map_score = label_ranking_average_precision_score(augment_true_relevance, score)
#         # print(true_relevance, score, map_score)
#         augment_map_scores.append(round(map_score, 4))

#     augment_map_score = np.mean(augment_map_scores)
#     print(augment_map_score)
#     metrics_df["augment_map_score"] = augment_map_scores

#     return metrics_df


# # metrics_df = run_metrics(score_df)
# # metrics_df.to_csv("scores/AN-metrics.csv")
# # metrics_df.to_excel("scores/AN-metrics.xlsx")

# # 0.3946842857142857
# # 0.11904285714285713
# # 0.84722
