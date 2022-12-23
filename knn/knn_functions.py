import pandas as pd
import math as m
import numpy as np
import statistics as s
# from sklearn.metrics import f1_score
import sklearn as sk

distance_type = "euclidean"
kernel_f_type = "sigmoid"

kernel_functions = {
    "uniform": lambda u: 0.5 if abs(u) < 1 else 0,
    "triangular": lambda u: 1.0 - abs(u) if abs(u) < 1 else 0,
    "epanechnikov": lambda u: (3.0 / 4.0) * pow(1.0 - u * u, 1) if abs(u) < 1 else 0,
    "quartic": lambda u: (15.0 / 16.0) * pow(1.0 - u * u, 2) if abs(u) < 1 else 0,
    "triweight": lambda u: (35.0 / 32.0) * pow(1.0 - u * u, 3) if abs(u) < 1 else 0,
    "tricube": lambda u: (70.0 / 81.0) * pow(1.0 - abs(u * u * u), 3) if abs(u) < 1 else 0,
    "cosine": lambda u: (m.pi / 4.0) * m.cos(u * m.pi / 2.0) if abs(u) < 1 else 0,

    "gaussian": lambda u: pow(m.e, -0.5 * u * u) / m.sqrt(2.0 * m.pi),
    "logistic": lambda u: 1.0 / (pow(m.e, u) + 2.0 + pow(m.e, -u)),
    "sigmoid": lambda u: (2.0 / m.pi) * (1.0 / (pow(m.e, u) + pow(m.e, -u)))
}

distance_functions = {
    "manhattan": lambda x, y: np.linalg.norm(np.subtract(x, y), 1),
    "euclidean": lambda x, y: np.linalg.norm(np.subtract(x, y), 2),
    "chebyshev": lambda x, y: max(np.subtract(x, y))
}


def distance(x, y):
    return distance_functions[distance_type](x, y)


def kernel_function(u):
    return kernel_functions[kernel_f_type](u)


def read_dataset(filename):
    dataframe = pd.read_csv(filename)
    dataframe.columns = ["F1", "F2", "F3", "F4", "F5", "Label"]
    return dataframe


def min_max(dataset):
    result = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        result.append([value_min, value_max])
    return result


#
def normalize(dataframe, minmax):
    # Normalizing dataframe values.
    # Last column "Labels" is skipped.
    for i, feature in enumerate(list(dataframe)[:-1]):
        dataframe[feature] = dataframe[feature].apply(lambda x: (x - minmax[i][0]) / (minmax[i][1] - minmax[i][0]))


def one_hot(dataframe):
    # Returns dataframe with encoded categorical features using one-hot encoding.
    # Numerical and label columns are moved to the end of dataframe (2 columns in my case).
    columns = list(dataframe)
    dataframe_with_dummies = pd.get_dummies(dataframe, columns=columns[:-2])
    columns = list(dataframe_with_dummies)
    return dataframe_with_dummies[np.append(columns[2:], columns[0:2])]


def predict_class(x_test, x_train, y_train, h, k=0):
    # List of pairs: (label, distance between x_test and x_with_label.
    distances = sorted([(y, distance(x, x_test)) for y, x in zip(y_train, x_train)], key=lambda x: x[1])
    # If k != 0 find distance to closets k+1'th neighbour
    if k != 0:
        h = distances[k][1]

    if h != 0:
        up = sum([d[0] * kernel_function(d[1] / h) for d in distances])
        down = sum([kernel_function(d[1] / h) for d in distances])
        # If denominator is zero, return mean value of labels.
        return up / down if down != 0 else s.mean(y_train)
    else:
        closest = list(filter(lambda d: d[1] == 0, distances))
        return s.mean(map(lambda d: d[0], closest)) if len(closest) != 0 else s.mean(y_train)


def leave_one_out(dataframe, actual, predicted, h, k):
    x = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    for i in range(len(dataframe)):
        x_train = x.drop(i).to_numpy().tolist()
        y_train = y.drop(i).to_numpy().tolist()
        x_test = x.iloc[i].to_numpy().tolist()
        y_test = y.iloc[i].item()

        actual.append(y_test)
        predicted.append(predict_class(x_test, x_train, y_train, h, k))


def count_f_score(dataframe, max_label, h, k=0):
    cm = np.zeros((max_label, max_label))
    actual = list()
    predicted = list()
    leave_one_out(dataframe, actual, predicted, h, k)
    predicted = list(map(lambda z: max(min(max_label, round(z)), 1), predicted))
    for a, p in zip(actual, predicted):
        cm[p - 1][a - 1] += 1

    cm_rows = np.sum(cm, axis=1)
    cm_cols = np.sum(cm, axis=0)
    f_scores = list()
    for i, x in enumerate(cm.diagonal()):
        precision = x / cm_rows[i] if cm_rows[i] != 0 else 0
        recall = x / cm_cols[i] if cm_cols[i] != 0 else 0
        f_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0)

    f1_score_macro = s.mean(f_scores)
    f1_score_micro = sum(cm.diagonal()) / np.sum(cm)

    # Calculating F-macro and F-micro score based on http://neerc.ifmo.ru/wiki definition
    # cm_all = np.sum(cm)
    # precw = sum([x * cm_rows[i] / cm_cols[i] if cm_cols[i] != 0 else 0 for i, x in enumerate(cm.diagonal())]) / cm_all
    # recw = sum(cm.diagonal()) / cm_all
    # f1_score_macro = 2 * precw * recw / (precw + recw)
    # f1_score_micro = sum([x * cm_rows[i] for i, x in enumerate(f_scores)]) / np.sum(cm)

    # print("F1-scores:sklearn = ", f1_score(actual, predicted, average=None))
    # print("F1-scores:by-hand = ", f_scores)
    #
    # print("F1-score:macro_sklearn = ", f1_score(actual, predicted, average="macro"))
    # print("F1-score:macro_by-hand = ", f1_score_macro)
    #
    # print("F1-score:micro_sklearn = ", f1_score(actual, predicted, average="micro"))
    # print("F1-score:micro_by-hand = ", f1_score_micro)
    return f1_score_macro
