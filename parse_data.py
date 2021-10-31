import pandas as pd
import re
import json 
import emoji
import functools
import operator

import emoji
from emoji import unicode_codes
import random

import requests 
import bs4 

random.seed(10)

def google_definition(query):
    searchname = query.replace(' ', '+')
    url = "https://google.com/search?q=meaning+" + searchname
    request_result = requests.get( url )
    soup = bs4.BeautifulSoup( request_result.text, "html.parser" )
    heading_object=soup.find_all( "div" , class_='BNeawe s3v9rd AP7Wnd' )
    definition = heading_object[2].getText()
    return definition


def get_rand_emojis(len_emojis, exclude_emojis_text):
    # print(exclude_emojis_text)
    lang_keys = list(unicode_codes.EMOJI_UNICODE.keys())
    emoji_keys = list(unicode_codes.EMOJI_UNICODE[lang_keys[0]].keys())
    num_emojis = len(unicode_codes.EMOJI_UNICODE[lang_keys[0]])

    rand_emojis = []
    rand_emojis_text = []
    for i in range(0, len_emojis):
        rand_emoji_idx = random.randrange(0, num_emojis)
        rand_emoji_text = emoji_keys[rand_emoji_idx]
        while rand_emoji_text in exclude_emojis_text:
            rand_emoji_idx = random.randrange(0, num_emojis)
            rand_emoji_text = emoji_keys[rand_emoji_idx]
        rand_emoji = unicode_codes.EMOJI_UNICODE[lang_keys[0]][emoji_keys[rand_emoji_idx]]
        rand_emojis.append(rand_emoji)
        rand_emojis_text.append(rand_emoji_text)
    rand_emojis = " ".join(rand_emojis)
    # print(rand_emojis, rand_emojis_text)
    return rand_emojis, rand_emojis_text



# load attribute df from csv file
an_df = pd.read_csv("data/AN/an_attribute_ground_truth.csv")
col_names = an_df.columns.values
an_dict = dict( zip(an_df["concept"], an_df["attribute"]))

# Read all sheets directly into an ordered dictionary.
sheet_to_df_map = pd.read_excel("data/AN/an_data_collection.xlsx", sheet_name=None)
sheet_names = list( sheet_to_df_map.keys() )
sheet_names.remove('forms')

concept_count = 0
data_dict = {}

sheet_ids = [i for i in range(1, 26)]
print(sheet_ids)
for sheet_id in sheet_ids:
    sheet_name = str(sheet_id)
    print(sheet_name)
    
    sheet1 = sheet_to_df_map[sheet_name]
    col = sheet1.columns.values
    num_concepts = int ( ( len(col) - 6 ) / 3 )
    concept_count += num_concepts
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
            em_split = [emoji.demojize(em) for em in em_split]
            emojis_text_list.append(em_split)
        emoji_annotations_text_dict[concept_key] = emojis_text_list
    # print(emoji_annotations_text_dict)


    # to get baseline emojis for each concept
    baselines = ratings.columns.tolist()
    baselines = [[' '.join(re.split('\s+|\.|\t',s)[2:])] for s in baselines]
    baseline_dict = dict( zip(concepts, baselines) )

    ## convert baseline emojis to text representation
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
            em_split = [emoji.demojize(em) for em in em_split]
            baselines_text_list.append(em_split)
        baseline_text_dict[concept_key] = baselines_text_list
    

    ## construct semi-hard negative samples from the baseline emojis of the same adjective
    semineg_emoji_dict = {}
    for concept_key in baseline_dict:
        adj = concept_key.split()[0]
        semineg_emoji_list = []
        for curr_concept in baseline_dict:
            curr_adj = curr_concept.split()[0]
            if curr_concept == concept_key: # or curr_adj != adj 
                continue
            semineg_emoji_list += baseline_dict[curr_concept]
        semineg_emoji_dict[concept_key] = semineg_emoji_list
    ## convert semi-hard negative emojis to text representation
    semineg_text_dict = {}
    for concept_key in semineg_emoji_dict:
        semineg_text_list = []
        for emojis in semineg_emoji_dict[concept_key]:
            if type(emojis) != type('str'):
                # print(emojis)
                emojis_text_list.append(emojis)
                continue
            # print(emojis, type(emojis))
            em_split_emoji = emoji.get_emoji_regexp().split(emojis)
            em_split_whitespace = [substr.split() for substr in em_split_emoji]
            em_split = functools.reduce(operator.concat, em_split_whitespace)
            em_split = [emoji.demojize(em) for em in em_split]
            semineg_text_list.append(em_split)
        semineg_text_dict[concept_key] = semineg_text_list

    ## construct random negative samples, excluding emojis from baseline(hard negative) emojis and semi-hard negative emojis
    exclude_emojis_text = [semineg+baseline for (semineg, baseline) in zip(list(semineg_text_dict.values()),list(baseline_text_dict.values()))][0]
    exclude_emojis_text = [item for sublist in exclude_emojis_text for item in sublist]
    # print(exclude_emojis_text)
    randneg = [get_rand_emojis(len_emojis=2, exclude_emojis_text= exclude_emojis_text) for concept in concepts]
    randneg_emojis = list(list(zip(*randneg))[0])
    randneg_emojis_text = [[i] for i in list(list(zip(*randneg))[1])]
    randneg_emoji_dict = dict(zip(concepts, randneg_emojis))
    randneg_text_dict = dict(zip(concepts, randneg_emojis_text))

    # convert df to dict of list
    ratings = ratings.set_axis(concepts, axis='columns')
    ratings_dict = ratings.to_dict(orient='list')
    # print(ratings_dict)


    # combine (attribute, emoji sequence, ratings and negative sample) of each concept into single dict
    for concept in concepts:
        temp = {}
        temp['form_id'] = sheet_name
        temp['attribute'] = an_dict[concept]  # ground truth attribute
        temp['definition'] = google_definition(concept)
        temp['attribute_annotations'] = attribute_annotations_dict[concept]
        temp['rating_annotations'] = ratings_dict[concept]
        temp['emoji_annotations'] = emoji_annotations_dict[concept]
        temp['emoji_annotations_text'] = emoji_annotations_text_dict[concept]
        temp['baseline_emoji'] = baseline_dict[concept]
        temp['baseline_text'] = baseline_text_dict[concept]
        temp['semineg_emoji'] = semineg_emoji_dict[concept]
        temp['semineg_text'] = semineg_text_dict[concept]
        temp['randneg_emoji'] = randneg_emoji_dict[concept]
        temp['randneg_text'] = randneg_text_dict[concept]
        data_dict[concept] = temp
        # print(temp)
    # if concept_count > 2:
    #     break

print(concept_count)
# print(data_dict)

# save to json file
with open("data/AN/an-data-collection1.json", "w") as outfile:
    json.dump(data_dict, outfile, indent = 4, allow_nan = True) 


# load dict from json file
# f = open("data/data.json")
# json_object = json.load(f)
# print(json_object)
