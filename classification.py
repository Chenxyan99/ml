import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
mnist.target = mnist.target.astype(np.int8)
sort_by_target(mnist)

X, y = mnist["data"], mnist["target"]
some_digit = X[36000]


# 画着玩
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


# 划分训练集/测试集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 训练二分类器(数字5检测器)
# y_train_5 = (y_train == 5)
# t_test_5 = (y_test == 5)
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)

# 性能评估(交叉验证)
# skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]
#
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))

# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# 混淆矩阵
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(confusion_matrix(y_train_5, y_train_pred))

# 准确率与召回率
# print(precision_score(y_train_5, y_train_pred))
# print(recall_score(y_train_5, y_train_pred))

# F1值(准确率和召回率的调和平均)
# print(f1_score(y_train_5, y_train_pred))

# y_scores = sgd_clf.decision_function([some_digit])
# print(y_scores)
# thresholds = [0, 200000]
# for threshold in thresholds:
#     y_some_digit_pred = (y_scores > threshold)
#     print(y_some_digit_pred)

# 准确率和召回率的折衷
# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# if y_scores.ndim == 2:
#     y_scores = y_scores[:, 1]

# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
#     plt.xlabel("Threshold")
#     plt.legend(loc="upper left")
#     plt.ylim([0, 1])


# plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
# plt.show()

# y_train_pred_90 = (y_scores > 6000)
# print(precision_score(y_train_5, y_train_pred_90))
# print(recall_score(y_train_5, y_train_pred_90))

# ROC曲线
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")


# plot_roc_curve(fpr, tpr)
# plt.show()

# print(roc_auc_score(y_train_5, y_scores))

# forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=5, method="predict_proba")
# y_scores_forest = y_probas_forest[:, 1]
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
# plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right")
# plt.show()
# print(roc_auc_score(y_train_5, y_scores_forest))

# 多类分类
# sgd_clf.fit(X_train, y_train)
# sgd_clf.predict([some_digit])
#
# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)

# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# ovo_clf.predict([some_digit])
# print(len(ovo_clf.estimators_))

# forest_clf.fit(X_train, y_train)
# forest_clf.predict([some_digit])
# print(forest_clf.predict_proba([some_digit]))

# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# 误差分析
# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()
#
# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.figure(figsize=(8,8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()

# 多标签分类
# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)
# print(knn_clf.predict([some_digit]))
#
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
# print(f1_score(y_train, y_train_knn_pred, average="macro"))

# 多标签输出分类
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index = 5500
plt.subplot(121); plot_digits(X_test_mod[some_index])
plt.subplot(122); plot_digits(y_test_mod[some_index])
plt.show()

# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[some_index]])
# plot_digits(clean_digit)
# plt.show()