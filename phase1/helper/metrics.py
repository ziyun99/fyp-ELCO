# import numpy as np
# from sklearn.metrics import label_ranking_average_precision_score
# y_true = np.array([[1, 0, 0], [0, 0, 1]])
# y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
# score = label_ranking_average_precision_score(y_true, y_score)
# print(score)

# from sklearn.metrics import label_ranking_average_precision_score
# y_true = np.array([[1, 0, 0], [0, 0, 1]])
# y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
# score =label_ranking_average_precision_score(y_true, y_score)
# print(score)
# y_true = np.array([[1, 0, 0]])
# y_score = np.array([[0.75, 0.5, 1]])
# score = label_ranking_average_precision_score(y_true, y_score)
# print(score)
# y_true = np.array([[0, 0, 1]])
# y_score = np.array([[1, 0.2, 0.1]])
# score = label_ranking_average_precision_score(y_true, y_score)
# print(score)

# from sklearn.metrics import label_ranking_loss
# y_true = np.array([[1, 0, 0], [0, 0, 1]])
# y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
# score =label_ranking_loss(y_true, y_score)
# print(score)
# y_true = np.array([[1, 0, 0]])
# y_score = np.array([[0.75, 0.5, 1]])
# score = label_ranking_loss(y_true, y_score)
# print(score)
# y_true = np.array([[0, 0, 1]])
# y_score = np.array([[1, 0.2, 0.1]])
# score = label_ranking_loss(y_true, y_score)
# print(score)

# # With the following prediction, we have perfect and minimal loss
# y_true = np.array([[1, 0, 0], [0, 0, 1]])
# y_score = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
# score =label_ranking_loss(y_true, y_score)
# print(score)


import numpy as np
from sklearn.metrics import ndcg_score

y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
score =ndcg_score(y_true, y_score)
print(score)
y_true = np.array([[1, 0, 0]])
y_score = np.array([[0.75, 0.5, 1]])
score = ndcg_score(y_true, y_score)
print(score)
y_true = np.array([[1, 0, 0]])
y_score = np.array([[0.9, 0.2, 1]])
score = ndcg_score(y_true, y_score)
print(score)
y_true = np.array([[0, 0, 1]])
y_score = np.array([[1, 0.2, 0.1]])
score = ndcg_score(y_true, y_score)
print(score)

# we have groud-truth relevance of some answers to a query:
true_relevance = np.asarray([[10, 0, 0, 1, 5]])
# we predict some scores (relevance) for the answers
scores = np.asarray([[.1, .2, .3, 4, 70]])
ndcg_score(true_relevance, scores)

scores = np.asarray([[.05, 1.1, 1., .5, .0]])
ndcg_score(true_relevance, scores)

# we can set k to truncate the sum; only top k answers contribute.
ndcg_score(true_relevance, scores, k=4)

# the normalization takes k into account so a perfect answer
# would still get 1.0
ndcg_score(true_relevance, true_relevance, k=4)

# now we have some ties in our prediction
scores = np.asarray([[1, 0, 0, 0, 1]])
# by default ties are averaged, so here we get the average (normalized)
# true relevance of our top predictions: (10 / 10 + 5 / 10) / 2 = .75
ndcg_score(true_relevance, scores, k=1)

# we can choose to ignore ties for faster results, but only
# if we know there aren't ties in our scores, otherwise we get
# wrong results:
ndcg_score(true_relevance,
          scores, k=1, ignore_ties=True)
