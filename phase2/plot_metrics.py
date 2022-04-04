import sys

import matplotlib.pyplot as plt
import pandas as pd


model_file = sys.argv[1]
k = 200
df_mse_train = pd.read_csv(f'output/multilingual/{model_file}/eval/mse_evaluation_mse-train_results.csv')[:k]
df_mse_test = pd.read_csv(f'output/multilingual/{model_file}/eval/mse_evaluation_mse-test_results.csv')[:k]
df_mse_val = pd.read_csv(f'output/multilingual/{model_file}/eval/mse_evaluation_mse-validate_results.csv')[:k]
df_sim_train = pd.read_csv(f'output/multilingual/{model_file}/eval/similarity_evaluation_cossim-train_results.csv')[:k]
df_sim_test = pd.read_csv(f'output/multilingual/{model_file}/eval/similarity_evaluation_cossim-test_results.csv')[:k]
df_sim_val = pd.read_csv(f'output/multilingual/{model_file}/eval/similarity_evaluation_cossim-validate_results.csv')[:k]
df_trans_train = pd.read_csv(f'output/multilingual/{model_file}/eval/translation_evaluation_trans-train_results.csv')[:k]
df_trans_test = pd.read_csv(f'output/multilingual/{model_file}/eval/translation_evaluation_trans-test_results.csv')[:k]
df_trans_val = pd.read_csv(f'output/multilingual/{model_file}/eval/translation_evaluation_trans-validate_results.csv')[:k]

f1 = plt.figure(figsize=(30,10))
ax_mse = f1.add_subplot(3,2,1)
ax_mse.title.set_text("MSE")
ax_mse.plot(df_mse_test[["MSE"]], label = 'test', color = "r")
ax_mse.plot(df_mse_val[["MSE"]], label = 'val', color="g")
ax_mse.plot(df_mse_train[["MSE"]], label = 'train', color = "b")
ax_mse.legend(loc="upper right")

ax_sim = f1.add_subplot(3,2,3)
ax_sim.title.set_text("cos_sim-cosine_pearson")
ax_sim.plot(df_sim_test[["cosine_pearson"]], label = 'test', color = "r")
ax_sim.plot(df_sim_val[["cosine_pearson"]], label = 'val', color="g")
ax_sim.plot(df_sim_train[["cosine_pearson"]], label = 'train', color = "b")
ax_sim.legend(loc="upper right")

ax_sim = f1.add_subplot(3,2,4)
ax_sim.title.set_text("cos_sim-cosine_spearman")
ax_sim.plot(df_sim_test[["cosine_spearman"]], label = 'test', color = "r")
ax_sim.plot(df_sim_val[["cosine_spearman"]], label = 'val', color="g")
ax_sim.plot(df_sim_train[["cosine_spearman"]], label = 'train', color = "b")
ax_sim.legend(loc="upper right")

ax_trans = f1.add_subplot(3,2,5)
ax_trans.title.set_text("translation_src2trg")
ax_trans.plot(df_trans_test[["src2trg"]], label = 'test', color = "r")
ax_trans.plot(df_trans_val[["src2trg"]], label = 'val', color="g")
ax_trans.plot(df_trans_train[["src2trg"]], label = 'train', color="b")
ax_trans.legend(loc="upper right")

ax_trans = f1.add_subplot(3,2,6)
ax_trans.title.set_text("translation_trg2src")
ax_trans.plot(df_trans_test[["trg2src"]], label = 'test', color = "r")
ax_trans.plot(df_trans_val[["trg2src"]], label = 'val', color="g")
ax_trans.plot(df_trans_train[["trg2src"]], label = 'train', color="b")
ax_trans.legend(loc="upper right")


f1.savefig(f"output/multilingual/{model_file}/eval/plot-200.jpg")



df_infroret_test = pd.read_csv(f'output/multilingual/{model_file}/eval/Information-Retrieval_evaluation_inforet-test_results.csv')[:k]
f2 = plt.figure(figsize=(30,10))
ax_infroret = f2.add_subplot(2,2,1)
ax_infroret.title.set_text("cos_sim-Accuracy")
ax_infroret.plot(df_infroret_test[["cos_sim-Accuracy@1"]], label = 'Accuracy@1', color = "r")
ax_infroret.plot(df_infroret_test[["cos_sim-Accuracy@3"]], label = 'Accuracy@3', color="g")
ax_infroret.plot(df_infroret_test[["cos_sim-Accuracy@5"]], label = 'Accuracy@5', color="b")
ax_infroret.plot(df_infroret_test[["cos_sim-Accuracy@10"]], label = 'Accuracy@10', color="purple")
ax_infroret.legend(loc="upper right")

ax_infroret = f2.add_subplot(2,2,2)
ax_infroret.title.set_text("cos_sim-Precision")
ax_infroret.plot(df_infroret_test[["cos_sim-Precision@1"]], label = 'Precision@1', color = "r")
ax_infroret.plot(df_infroret_test[["cos_sim-Precision@3"]], label = 'Precision@3', color="g")
ax_infroret.plot(df_infroret_test[["cos_sim-Precision@5"]], label = 'Precision@5', color="b")
ax_infroret.plot(df_infroret_test[["cos_sim-Precision@10"]], label = 'Precision@10', color="purple")
ax_infroret.legend(loc="upper right")

ax_infroret = f2.add_subplot(2,2,3)
ax_infroret.title.set_text("cos_sim-Recall")
ax_infroret.plot(df_infroret_test[["cos_sim-Recall@1"]], label = 'Recall@1', color = "r")
ax_infroret.plot(df_infroret_test[["cos_sim-Recall@3"]], label = 'Recall@3', color="g")
ax_infroret.plot(df_infroret_test[["cos_sim-Recall@5"]], label = 'Recall@5', color="b")
ax_infroret.plot(df_infroret_test[["cos_sim-Recall@10"]], label = 'Recall@10', color="purple")
ax_infroret.legend(loc="upper right")

ax_infroret = f2.add_subplot(2,2,4)
ax_infroret.title.set_text("cos_sim")
ax_infroret.plot(df_infroret_test[["cos_sim-MRR@10"]], label = 'MRR@10', color = "r")
ax_infroret.plot(df_infroret_test[["cos_sim-NDCG@10"]], label = 'NDCG@10', color="g")
ax_infroret.plot(df_infroret_test[["cos_sim-MAP@100"]], label = 'MAP@100', color="b")
ax_infroret.legend(loc="upper right")


f2.savefig(f"output/multilingual/{model_file}/eval/plot-inforet-200.jpg")


df_infroret_test = pd.read_csv(f'output/multilingual/{model_file}/eval/Information-Retrieval_evaluation_inforet-test_results.csv')[:k]
f2 = plt.figure(figsize=(30,10))
ax_infroret = f2.add_subplot(2,2,1)
ax_infroret.title.set_text("dot_score-Accuracy")
ax_infroret.plot(df_infroret_test[["dot_score-Accuracy@1"]], label = 'Accuracy@1', color = "r")
ax_infroret.plot(df_infroret_test[["dot_score-Accuracy@3"]], label = 'Accuracy@3', color="g")
ax_infroret.plot(df_infroret_test[["dot_score-Accuracy@5"]], label = 'Accuracy@5', color="b")
ax_infroret.plot(df_infroret_test[["dot_score-Accuracy@10"]], label = 'Accuracy@10', color="purple")
ax_infroret.legend(loc="upper right")

ax_infroret = f2.add_subplot(2,2,2)
ax_infroret.title.set_text("dot_score-Precision")
ax_infroret.plot(df_infroret_test[["dot_score-Precision@1"]], label = 'Precision@1', color = "r")
ax_infroret.plot(df_infroret_test[["dot_score-Precision@3"]], label = 'Precision@3', color="g")
ax_infroret.plot(df_infroret_test[["dot_score-Precision@5"]], label = 'Precision@5', color="b")
ax_infroret.plot(df_infroret_test[["dot_score-Precision@10"]], label = 'Precision@10', color="purple")
ax_infroret.legend(loc="upper right")

ax_infroret = f2.add_subplot(2,2,3)
ax_infroret.title.set_text("dot_score-Recall")
ax_infroret.plot(df_infroret_test[["dot_score-Recall@1"]], label = 'Recall@1', color = "r")
ax_infroret.plot(df_infroret_test[["dot_score-Recall@3"]], label = 'Recall@3', color="g")
ax_infroret.plot(df_infroret_test[["dot_score-Recall@5"]], label = 'Recall@5', color="b")
ax_infroret.plot(df_infroret_test[["dot_score-Recall@10"]], label = 'Recall@10', color="purple")
ax_infroret.legend(loc="upper right")

ax_infroret = f2.add_subplot(2,2,4)
ax_infroret.title.set_text("dot_score")
ax_infroret.plot(df_infroret_test[["dot_score-MRR@10"]], label = 'MRR@10', color = "r")
ax_infroret.plot(df_infroret_test[["dot_score-NDCG@10"]], label = 'NDCG@10', color="g")
ax_infroret.plot(df_infroret_test[["dot_score-MAP@100"]], label = 'MAP@100', color="b")
ax_infroret.legend(loc="upper right")


f2.savefig(f"output/multilingual/{model_file}/eval/plot-inforet-dot_score-200.jpg")
