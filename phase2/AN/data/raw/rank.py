import json
import pandas as pd

f = open('AN_scoring_mpnet.json')
data_dict = json.load(f)


for concept in data_dict:
    rank_rand_neg = data_dict[concept]["scores_ranked"][0][0] 
    rank_rand_neg_augment = data_dict[concept]["scores_ranked"][1][0] 

    if rank_rand_neg_augment == 3:
        print("\nrand neg")
        print(concept)
        print(data_dict[concept]["scores_ranked"])
        print(data_dict[concept]["emoji_annotations"])
        print(data_dict[concept]["randneg_emoji"])

    rank_baseline = data_dict[concept]["scores_ranked"][0][2] 
    rank_baseline_augment = data_dict[concept]["scores_ranked"][1][2] 

    # if rank_baseline != 3 and rank_baseline_augment == 3:
    #     print("\nbaseline")
    #     print(concept)
    #     print(data_dict[concept]["scores_ranked"])
    #     print(data_dict[concept]["emoji_annotations"])
    #     print(data_dict[concept]["baseline_emoji"])

    rank_pos = data_dict[concept]["scores_ranked"][0][3] 
    rank_pos_augment = data_dict[concept]["scores_ranked"][1][3] 

    # if rank_pos == 3 and rank_pos_augment != 3:
    #     print("\npos")
    #     print(concept)
    #     print(data_dict[concept]["scores_ranked"])
    #     print(data_dict[concept]["emoji_annotations"])
    #     print(data_dict[concept]["baseline_emoji"])