import pandas as pd
import json
from collections import Counter
import pprint

# convert scores into ranking of each samples: ideal ranking is [0(randneg), 1(semineg), 2(baseline), 3(positive)]
# return ranking of those without & with semineg samples, each [ [without augmentation], [with augmentation] ].
def get_score_ranking(filepath):
    f = open(filepath)
    data_dict = json.load(f)

    rankings = []
    semineg_rankings = []

    for concept in data_dict:
        # drop the semineg scores
        scores = [[s[0]] + s[2:] for s in data_dict[concept]["scores"]]
        ranking = [
            [sorted(list(set(score))).index(x) for x in score] for score in scores
        ]
        rankings.append(ranking)

        semineg_scores = data_dict[concept]["scores"]
        semineg_ranking = [
            [sorted(list(set(score))).index(x) for x in score]
            for score in semineg_scores
        ]
        semineg_rankings.append(semineg_ranking)

    return rankings, semineg_rankings


def ranking_evaluation(rankings, category_outcome_dict, category_set_name):
    len_dataset = len(rankings)
    print(len_dataset)

    dataset_type_dict = {0: "raw", 1: "augmented"}

    for dataset_idx, dataset_type in dataset_type_dict.items():
        category_max = []
        for row in rankings:
            for (rank_category, rank_num) in enumerate(row[dataset_idx]):
                if rank_num == max(row[0]):
                    category_max.append(rank_category)
        counter = dict(Counter(category_max))

        for category_id, max_count in counter.items():
            category_outcome_dict[category_set_name][dataset_type][category_id][
                "max_count"
            ] = max_count
            category_outcome_dict[category_set_name][dataset_type][category_id][
                "max_count_percentage"
            ] = (max_count / len_dataset)

        print(
            f"  Dataset: {dataset_type} \n  Ranking evaluation outcome: \n \
        - Number of samples ranked 1st: {counter} \n"
        )

    return category_outcome_dict


def evaluate(filepath):
    print(f"\n=> Ranking evaluation on: {filepath}\n")

    category_set_names = ["without semineg samples", "with semineg samples"]
    category_set_dict = {
        category_set_names[0]: {0: "randneg", 1: "baseline", 2: "positive"},
        category_set_names[1]: {
            0: "randneg",
            1: "semineg",
            2: "baseline",
            3: "positive",
        },
    }
    category_outcome_dict = {
        category_set_names[0]: {
            "raw": {
                0: {"category_name": "randneg"},
                1: {"category_name": "baseline"},
                2: {"category_name": "positive"},
            },
            "augmented": {
                0: {"category_name": "randneg"},
                1: {"category_name": "baseline"},
                2: {"category_name": "positive"},
            },
        },
        category_set_names[1]: {
            "raw": {
                0: {"category_name": "randneg"},
                1: {"category_name": "semineg"},
                2: {"category_name": "baseline"},
                3: {"category_name": "positive"},
            },
            "augmented": {
                0: {"category_name": "randneg"},
                1: {"category_name": "semineg"},
                2: {"category_name": "baseline"},
                3: {"category_name": "positive"},
            },
        },
    }

    print("Category sets:")
    for category_set_name, category_ref in category_set_dict.items():
        str = pprint.pformat(category_ref)
        print(f" -> {category_set_name}: {str}")

    # convert sbert-similarity-scores to rankings within categories
    rankings, semineg_rankings = get_score_ranking(filepath)

    # define the two settings of categories
    category_set_rankings = {
        category_set_names[0]: rankings,
        category_set_names[1]: semineg_rankings,
    }
    print()

    # evaluate the rankings
    for category_set_name in category_set_rankings:
        print(f"Category set: {category_set_name}")
        category_outcome = ranking_evaluation(
            category_set_rankings[category_set_name],
            category_outcome_dict,
            category_set_name,
        )

    pprint.pprint(category_outcome)


filepath = "../data/AN_scoring_mpnet.json"
evaluate(filepath)

#     for i in [0, 1]:
#         print(i)
#         randneg = [1 for rank in ranks if rank[i][0] == 2]
#         print("randneg", len(randneg))

#         baseline = [1 for rank in ranks if rank[i][1] == 2]
#         print("baseline", len(baseline))

#         pos = [1 for rank in ranks if rank[i][2] == 2]
#         print("pos", len(pos))

#         randneg_baseline = [1 for rank in ranks if rank[i][0] > rank[i][1]]
#         print("randneg > baseline", len(randneg_baseline))

#         randneg_pos = [1 for rank in ranks if rank[i][0] > rank[i][2]]
#         print("randneg > pos", len(randneg_pos))

#         baseline_pos = [1 for rank in ranks if rank[i][1] > rank[i][2]]
#         print("baseline > pos", len(baseline_pos))

#     ranks = rank_semineg
#     for i in [0, 1]:
#         print(i)
#         randneg = [1 for rank in ranks if rank[i][0] == 3]
#         print("randneg", len(randneg))

#         semineg = [1 for rank in ranks if rank[i][1] == 3]
#         print("semineg", len(semineg))

#         baseline = [1 for rank in ranks if rank[i][2] == 3]
#         print("baseline", len(baseline))

#         pos = [1 for rank in ranks if rank[i][3] == 3]
#         print("pos", len(pos))


#         randneg_semineg = [1 for rank in ranks if rank[i][0] > rank[i][1]]
#         print("randneg > semineg", len(randneg_semineg))

#         randneg_baseline = [1 for rank in ranks if rank[i][0] > rank[i][2]]
#         print("randneg > baseline", len(randneg_baseline))

#         randneg_pos = [1 for rank in ranks if rank[i][0] > rank[i][3]]
#         print("randneg > pos", len(randneg_pos))

#         semineg_baseline = [1 for rank in ranks if rank[i][1] > rank[i][2]]
#         print("semineg > baseline", len(semineg_baseline))

#         semineg_pos = [1 for rank in ranks if rank[i][1] > rank[i][3]]
#         print("semineg > pos", len(semineg_pos))

#         baseline_pos = [1 for rank in ranks if rank[i][2] > rank[i][3]]
#         print("baseline > pos", len(baseline_pos))

# ranking_evaluation()

# # df = pd.read_csv("scores/AN-score.csv")
# # df = df.drop(columns=["semineg_avg", "augment_semineg_avg", "semineg_max", "augment_semineg_max", "pos_avg", "augment_pos_avg"])
# # print(df.shape)


# # # randneg
# # # semineg_max
# # # baseline
# # # pos_max

# # df2 = df.loc[(df['baseline'] > df['pos_max']) & (df['augment_pos_max'] > df['augment_baseline'])]
# # df1 = df.loc[(df['pos_max'] > df['baseline']) & (df['augment_baseline'] > df['augment_pos_max'])]
# # print(df1)
# # print(df2)
# # print(len(df1), len(df2))

# # s1 = pd.merge(df1, df2, how='inner')
# # print(s1)

# # set_diff_df = pd.concat([df2, df1, df1]).drop_duplicates(keep=False)
# # print(set_diff_df)

# target_col = 'adj'
# target='bright'

# df = pd.read_excel("scores/AN-score.xlsx")
# print(df.shape)
# win = df.loc[(df[target_col] == target) & (df['randneg'] > df['semineg_max']) & (df['randneg'] > df['baseline']) & (df['randneg'] > df['pos_max'])]
# print(len(win))

# win = df.loc[(df[target_col] == target) & (df['semineg_max'] > df['randneg']) & (df['semineg_max'] > df['baseline']) & (df['semineg_max'] > df['pos_max'])]
# print(len(win))

# win = df.loc[(df[target_col] == target) & (df['baseline'] > df['randneg']) & (df['baseline'] > df['semineg_max']) & (df['baseline'] > df['pos_max'])]
# print(len(win))

# win = df.loc[(df[target_col] == target) & (df['pos_max'] > df['randneg']) & (df['pos_max'] > df['semineg_max']) & (df['pos_max'] > df['baseline'])]
# print(len(win))
# # augment_randneg
# # augment_semineg_max
# # augment_baseline
# # augment_pos_max

# win = df.loc[(df[target_col] == target) & (df['augment_randneg'] > df['augment_semineg_max']) & (df['augment_randneg'] > df['augment_baseline']) & (df['augment_randneg'] > df['augment_pos_max'])]
# print(len(win))

# win = df.loc[(df[target_col] == target) & (df['augment_semineg_max'] > df['augment_randneg']) & (df['augment_semineg_max'] > df['augment_baseline']) & (df['augment_semineg_max'] > df['augment_pos_max'])]
# print(len(win))

# win = df.loc[(df[target_col] == target) & (df['augment_baseline'] > df['augment_randneg']) & (df['augment_baseline'] > df['augment_semineg_max']) & (df['augment_baseline'] > df['augment_pos_max'])]
# print(len(win))

# win = df.loc[(df[target_col] == target) & (df['augment_pos_max'] > df['augment_randneg']) & (df['augment_pos_max'] > df['augment_semineg_max']) & (df['augment_pos_max'] > df['augment_baseline'])]
# print(len(win))

# target_col = 'adj'
# target='domestic'
# win = df.loc[(df[target_col] == target)]
# print(win)

# # Attributes
# # total 77
# # randneg 0
# # baseline 10
# # pos_max 67

# # ATTRIBUTE-AUGMENT
# # total 77
# # randneg 0
# # baseline 10
# # pos_max 67

# # Adj
# # total 45
# # randneg 0
# # baseline 3
# # pos_max 42

# # ADJ-AUGMENT
# # total 45
# # randneg 0
# # baseline 3
# # pos_max 42
