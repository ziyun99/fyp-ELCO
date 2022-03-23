from os import link
from numpy import NaN
import pandas as pd
import numpy as np

seed = 27
np.random.seed(seed)

df = pd.read_excel("../registration_form.xlsx")
reserved_list0 = df.loc[(df["Num_samples"] < 5) | pd.isnull(df["Num_samples"])]
target_list0 = df.loc[df["Num_samples"] >= 5]


target_list1, reserved_list1 = np.split(target_list0.sample(frac=1, random_state=seed), [39])
target_list1 = pd.concat([target_list1, df.loc[df["Phone number"] == "98289667"]])
reserved_list1 = reserved_list1.drop(reserved_list1[reserved_list1["Phone number"] == "98289667"].index)
print(f"total participant size: {df.shape}, reserved list size(num_samples <= 4): {reserved_list0.shape}, target participant size: {target_list1.shape}, other reserved list size {reserved_list1.shape}")


### Assign survey forms to participant randomly ###
forms = pd.read_excel("../form_links.xlsx")
links = np.array(forms["link"].to_list())
passcodes = np.array(forms["passcode"].to_list())

all_link_id = [np.arange(1, 26) for _ in range(8)]
for link_id in all_link_id:
    np.random.shuffle(link_id)

link_id_stacks = []
for link_id in all_link_id:
    for j in range(0, 25, 5):
        # each participant is assigned 5 forms
        temp = link_id[j: j+5]
        link_id_stacks.append(temp)    

# stack into 2d array
link_id_stack = np.vstack(link_id_stacks)
link_stack = [[links[col-1] for col in row] for row in link_id_stack]
passcode_stack = [[passcodes[col-1] for col in row] for row in link_id_stack]

# convert to dataframe
link_id_df = pd.DataFrame(link_id_stack, columns=['form_id1', 'form_id2', 'form_id3', 'form_id4', 'form_id5'])
link_df = pd.DataFrame(link_stack, columns=['form_link1', 'form_link2', 'form_link3', 'form_link4', 'form_link5'])
passcode_df = pd.DataFrame(passcode_stack, columns=['passcode1', 'passcode2', 'passcode3', 'passcode4', 'passcode5'])

participant_df = pd.concat([target_list1.reset_index(), link_id_df, passcode_df, link_df], axis=1)
# participant_df.to_excel('../participant.xlsx')

reserved_df = pd.concat([reserved_list0.reset_index(), reserved_list1.reset_index()], axis=0)
reserved_df.to_excel('../reserved_list.xlsx')
print(reserved_df.shape)