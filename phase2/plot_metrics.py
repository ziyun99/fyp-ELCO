import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statistics as st
from sentence_transformers import SentenceTransformer, util
import numpy as np


# def plot_line_graph(score_df, col, img_name):
#     img_path = "figures/{}/{}_{}.png".format(dataset_name, dataset_name, img_name)
#     fig = px.line(score_df,  y=col)
#     fig.update_layout(xaxis_title="ZWJ concepts", yaxis_title="similarity scores", legend_title="Emoji sequences", margin = {'l':5,'r':5,'t':5,'b':5},)
#     fig.write_image(img_path, format="png", width=600, height=350, scale=2)

# df = pd.read_csv('output/multilingual/model-2022-02-17_06-45/eval/mse_evaluation_mse-test_results.csv')

# col = ['pos', 'augment_pos']
# img_name = "line_graph_pos"
# plot_line_graph(score_df, col, img_name)

import os
import pandas as pd
import matplotlib.pyplot as plt

model_file = 'model-2022-02-17_13-40'
df_mse_test = pd.read_csv(f'output/multilingual/{model_file}/eval/mse_evaluation_mse-test_results.csv')
df_mse_val = pd.read_csv(f'output/multilingual/{model_file}/eval/mse_evaluation_mse-validate_results.csv')
df_sim_test = pd.read_csv(f'output/multilingual/{model_file}/eval/similarity_evaluation_cossim-test_results.csv')
df_sim_val = pd.read_csv(f'output/multilingual/{model_file}/eval/similarity_evaluation_cossim-val_results.csv')
df_trans_test = pd.read_csv(f'output/multilingual/{model_file}/eval/translation_evaluation_trans-test_results.csv')
df_trans_val = pd.read_csv(f'output/multilingual/{model_file}/eval/translation_evaluation_trans-validate_results.csv')

f1 = plt.figure(figsize=(30,10))
ax_mse = f1.add_subplot(3,2,1)
ax_mse.title.set_text("MSE")
ax_mse.plot(df_mse_test[["MSE"]], label = 'test', color = "r")
ax_mse.plot(df_mse_val[["MSE"]], label = 'val', color="g")

ax_sim = f1.add_subplot(3,2,3)
ax_sim.title.set_text("cos_sim-cosine_pearson")
ax_sim.plot(df_sim_test[["cosine_pearson"]], label = 'test', color = "r")
ax_sim.plot(df_sim_val[["cosine_pearson"]], label = 'val', color="g")

ax_sim = f1.add_subplot(3,2,4)
ax_sim.title.set_text("cos_sim-cosine_spearman")
ax_sim.plot(df_sim_test[["cosine_spearman"]], label = 'test', color = "r")
ax_sim.plot(df_sim_val[["cosine_spearman"]], label = 'val', color="g")

ax_trans = f1.add_subplot(3,2,5)
ax_trans.title.set_text("translation_src2trg")
ax_trans.plot(df_trans_test[["src2trg"]], label = 'test', color = "r")
ax_trans.plot(df_trans_val[["src2trg"]], label = 'val', color="g")

ax_trans = f1.add_subplot(3,2,6)
ax_trans.title.set_text("translation_trg2src")
ax_trans.plot(df_trans_test[["trg2src"]], label = 'test', color = "r")
ax_trans.plot(df_trans_val[["trg2src"]], label = 'val', color="g")


f1.savefig(f"output/multilingual/{model_file}/eval/plot.jpg")