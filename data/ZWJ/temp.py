import pandas as pd
df = pd.read_csv("zwj-selected.csv")

df = df.drop(["unicode", "zwj_split_emojis"], 1)
print(df)

df.to_csv('zwj-selec.csv')