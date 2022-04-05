import pandas as pd

df = pd.read_csv("cluster_analysis_full.csv")
print(df.describe())

# pos = df.groupby('cluster')
# print(pos.describe())

counts = df['cluster'].value_counts()
print(counts)

df0 = df.loc[df["cluster"] == 0]
df1 = df.loc[df["cluster"] == 1]
df2 = df.loc[df["cluster"] == 2]
df3 = df.loc[df["cluster"] == 3]

df_list = [df0, df1, df2, df3]

for id, dfx in enumerate(df_list):
    print(f'Looking at Cluster {id}')
    # count by position
    # counts = dfx['pos'].value_counts()
    # print(counts)

    # count by

    print(dfx.describe())

# df0 = df0.sort_values(by=["dist_to_cluster_0"])
# df1 = df1.sort_values(by=["dist_to_cluster_1"])
# df2 = df2.sort_values(by=["dist_to_cluster_2"])
# df3 = df3.sort_values(by=["dist_to_cluster_3"])


# df = pd.concat([df0, df1, df2, df3])
# print(df)

# df0.to_csv("cluster_analysis_0.csv")
# df1.to_csv("cluster_analysis_1.csv")
# df2.to_csv("cluster_analysis_2.csv")
# df3.to_csv("cluster_analysis_3.csv")