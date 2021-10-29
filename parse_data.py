import pandas as pd
import re
import json 
import emoji
import functools
import operator

# Read all sheets directly into an ordered dictionary.
sheet_to_df_map = pd.read_excel("an_data_collection.xlsx", sheet_name=None)
sheet_names = list( sheet_to_df_map.keys() )
sheet_names.remove('forms')

total_concepts = 0
data_dict = {}

for sheet_name in sheet_names:
    print(sheet_name)
    
    sheet1 = sheet_to_df_map[sheet_name]
    col = sheet1.columns.values
    num_concepts = int ( ( len(col) - 6 ) / 3 )
    total_concepts += num_concepts
    # print(num_concepts)

    offset = 4

    start_idx = offset
    end_idx = offset + num_concepts 
    attribute_annotations = sheet1.iloc[:, start_idx: end_idx]

    start_idx = offset + num_concepts 
    end_idx = offset + num_concepts * 2 
    emoji_annotations = sheet1.iloc[:, start_idx: end_idx]

    start_idx = offset + num_concepts * 2
    end_idx = offset + num_concepts * 3 
    ratings = sheet1.iloc[:, start_idx: end_idx]

    # to get the list of AN concepts from column names
    concepts = emoji_annotations.columns.tolist()
    concepts = [' '.join(re.split('\s+|\.|\t',c)[:2]) for c in concepts]
    print(concepts)

    # convert df to dict of list
    attribute_annotations = attribute_annotations.set_axis(concepts, axis='columns')
    attribute_annotations_dict = attribute_annotations.to_dict(orient='list')
    # print(attribute_annotations_dict)

    # convert df to dict of list
    emoji_annotations = emoji_annotations.set_axis(concepts, axis='columns')
    emoji_annotations_dict = emoji_annotations.to_dict(orient='list')
    # print(emoji_annotations_dict)

    emoji_annotations_text_dict = {}
    for concept_key in emoji_annotations_dict:
        emojis_text_list = []
        for emojis in emoji_annotations_dict[concept_key]:
            if type(emojis) != type('str'):  # handle NaN cases
                # print(emojis)
                emojis_text_list.append(emojis)
                continue
            # print(emojis, type(emojis))
            em_split_emoji = emoji.get_emoji_regexp().split(emojis)
            em_split_whitespace = [substr.split() for substr in em_split_emoji]
            em_split = functools.reduce(operator.concat, em_split_whitespace)
            em_split = [emoji.demojize(em)[1:-1] for em in em_split]
            emojis_text_list.append(em_split)
        emoji_annotations_text_dict[concept_key] = emojis_text_list
    # print(emoji_annotations_text_dict)


    # to get the list of negative samples for each concept from column names
    baselines = ratings.columns.tolist()
    baselines = [[' '.join(re.split('\s+|\.|\t',s)[2:])] for s in baselines]
    baseline_dict = dict( zip(concepts, baselines) )
    # print(baselines)
    # print(baseline_dict)

    baseline_text_dict = {}
    for concept_key in baseline_dict:
        baselines_text_list = []
        for emojis in baseline_dict[concept_key]:
            if type(emojis) != type('str'):
                # print(emojis)
                emojis_text_list.append(emojis)
                continue
            # print(emojis, type(emojis))
            em_split_emoji = emoji.get_emoji_regexp().split(emojis)
            em_split_whitespace = [substr.split() for substr in em_split_emoji]
            em_split = functools.reduce(operator.concat, em_split_whitespace)
            em_split = [emoji.demojize(em)[1:-1] for em in em_split]
            baselines_text_list.append(em_split)
        baseline_text_dict[concept_key] = baselines_text_list
    

    # convert df to dict of list
    ratings = ratings.set_axis(concepts, axis='columns')
    ratings_dict = ratings.to_dict(orient='list')
    # print(ratings_dict)

    # combine (attribute, emoji sequence, ratings and negative sample) of each concept into single dict
    for concept in concepts:
        temp = {}
        temp['form_id'] = sheet_name
        temp['attribute_annotations'] = attribute_annotations_dict[concept]
        temp['emoji_annotations'] = emoji_annotations_dict[concept]
        temp['emoji_annotations_text'] = emoji_annotations_text_dict[concept]
        temp['rating_annotations'] = ratings_dict[concept]
        temp['baseline'] = baseline_dict[concept]
        temp['baseline_text'] = baseline_text_dict[concept]
        data_dict[concept] = temp
        # print(temp)
    # break

print(total_concepts)
# print(data_dict)


# convert dict to json
# json_object = json.dumps(data_dict, indent = 4, allow_nan = True) 
# print(json_object)

# save to json file
with open("data.json", "w") as outfile:
    json.dump(data_dict, outfile, indent = 4, allow_nan = True) 

# load dict from json file
f = open("data.json")
json_object = json.load(f)
# print(json_object)
