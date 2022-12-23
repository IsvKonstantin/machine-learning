import random
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier


def read_data(index):
    path_train = 'data/{:02d}_train.csv'.format(index)
    path_test = 'data/{:02d}_test.csv'.format(index)

    data_train = pd.read_csv(path_train).to_numpy()
    data_test = pd.read_csv(path_test).to_numpy()

    return data_train.T[0:-1].T, data_train.T[-1], data_test.T[0:-1].T, data_test.T[-1]


def accuracy_score(actual, predicted):
    score = sum([int(a == p) for a, p in zip(actual, predicted)]) / len(actual)
    return score


def find_best_params(index):
    data_train_x, data_train_y, data_test_x, data_test_y = read_data(index)
    best_accuracy = 0
    best_params = []

    depth_params = list(range(1, 15, 1))
    criterion_params = ['gini', 'entropy']
    splitter_params = ['best', 'random']

    for criterion in criterion_params:
        for splitter in splitter_params:
            for depth in depth_params:
                classifier = DecisionTreeClassifier(criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=depth)
                classifier.fit(data_train_x, data_train_y)

                actual = data_test_y
                predicted = classifier.predict(data_test_x)
                accuracy = accuracy_score(actual, predicted)

                if accuracy > best_accuracy:
                    best_params = [index, criterion, splitter, depth, accuracy]
                    best_accuracy = accuracy

    return best_params


def draw_graph(params, title):
    title = title + ', Dataset {:02d}'.format(params[0])
    data_train_x, data_train_y, data_test_x, data_test_y = read_data(params[0])
    train_accuracies = list()
    test_accuracies = list()
    depth_params = list(range(1, 20, 1))
    for depth in depth_params:
        classifier = DecisionTreeClassifier(criterion=params[1],
                                            splitter=params[2],
                                            max_depth=depth)
        classifier.fit(data_train_x, data_train_y)

        train_accuracies.append(accuracy_score(data_train_y, classifier.predict(data_train_x)))
        test_accuracies.append(accuracy_score(data_test_y, classifier.predict(data_test_x)))

    plt.plot(depth_params, train_accuracies, label='Train dataset')
    plt.plot(depth_params, test_accuracies, label='Test dataset')
    plt.xlabel('Height')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()


class DecisionForest:
    def __init__(self, n_estimators=20, should_reduce=False):
        self.n_estimators = n_estimators
        self.classifiers = list()
        self.reduced_features = list()
        self.should_reduce = should_reduce

    def prepare_data(self, data_x, data_y):
        new_data_x, new_data_y = list(), list()
        if self.should_reduce:
            random_features = random.sample(list(range(data_x.shape[1])), int(np.sqrt(data_x.shape[1])))
            random_features.sort()
        else:
            random_features = list(range(data_x.shape[1]))
        self.reduced_features.append(random_features)

        for _ in data_x:
            index = random.randint(0, data_x.shape[0] - 1)
            temp_row = list()
            for i in random_features:
                temp_row.append(data_x[index][i])

            new_data_x.append(temp_row)
            new_data_y.append(data_y[index])

        return np.array(new_data_x), np.array(new_data_y)

    def fit(self, data_x, data_y):
        for i in range(self.n_estimators):
            classifier = DecisionTreeClassifier()
            new_data_x, new_data_y = self.prepare_data(data_x, data_y)
            classifier.fit(new_data_x, new_data_y)
            self.classifiers.append(classifier)

    def predict_reduced(self, index, x):
        reduced_x = list()
        for i in self.reduced_features[index]:
            reduced_x.append(x[i])

        return self.classifiers[index].predict(reduced_x)

    def reduce_data(self, index, data):
        reduced_data = list()
        for row in data:
            reduced_row = list()
            for i in self.reduced_features[index]:
                reduced_row.append(row[i])
            reduced_data.append(reduced_row)
        return np.array(reduced_data)

    def predict(self, data_x):
        predictions = [defaultdict(lambda: 0) for _ in data_x]

        for index, classifier in enumerate(self.classifiers):
            reduced_data_x = self.reduce_data(index, data_x)
            predicted = classifier.predict(reduced_data_x)
            for i, p in enumerate(predicted):
                predictions[i][p] += 1

        answer = list()
        for p in predictions:
            answer.append(max(p, key=p.get))

        return answer


def print_forest_results():
    for index in range(1, 22, 1):
        data_train_x, data_train_y, data_test_x, data_test_y = read_data(index)
        classifier_1 = DecisionForest(20, False)
        classifier_1.fit(data_train_x, data_train_y)
        classifier_2 = DecisionForest(20, True)
        classifier_2.fit(data_train_x, data_train_y)
        accuracy_1 = accuracy_score(data_test_y, classifier_1.predict(data_test_x)) * 100
        accuracy_2 = accuracy_score(data_test_y, classifier_2.predict(data_test_x)) * 100
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Dataset {:02d}:'.format(index))
        print('random objects, n features      : {:.3}%'.format(accuracy_1))
        print('random objects, sqrt(n) features: {:.3}%'.format(accuracy_2))


# data_best_params = list()
# for i in range(1, 22, 1):
#     print(i)
#     data_best_params.append(find_best_params(i))
#
# data_best_params.sort(key=lambda a: a[3])
# data_max_depth = data_best_params[-1]
# data_min_depth = data_best_params[0]
#
# draw_graph(data_max_depth, 'Max height') # [21, 'entropy', 'best', 13, 0.8129]
# draw_graph(data_min_depth, 'Min height') # [3, 'gini, 'best', 1, 1]

# draw_graph([21, 'entropy', 'best', 13, 0.8129], 'Max height')
# draw_graph([5, 'gini', 'best', 1, 0.99], 'Min height')  # [3, 'gini, 'best', 1, 1]

draw_graph([21, 'entropy', 'best', 13, 0.8129], 'Max height')
draw_graph([3, 'gini', 'best', 1, 1], 'Min height')
draw_graph([5, 'gini', 'best', 1, 0.99], 'Min height')
print_forest_results()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 01:
# random objects, n features      : 99.6%
# random objects, sqrt(n) features: 73.1%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 02:
# random objects, n features      : 66.3%
# random objects, sqrt(n) features: 15.9%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 03:
# random objects, n features      : 99.0%
# random objects, sqrt(n) features: 86.5%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 04:
# random objects, n features      : 98.6%
# random objects, sqrt(n) features: 50.4%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 05:
# random objects, n features      : 99.6%
# random objects, sqrt(n) features: 92.9%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 06:
# random objects, n features      : 98.5%
# random objects, sqrt(n) features: 57.9%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 07:
# random objects, n features      : 98.5%
# random objects, sqrt(n) features: 42.4%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 08:
# random objects, n features      : 98.8%
# random objects, sqrt(n) features: 97.1%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 09:
# random objects, n features      : 82.2%
# random objects, sqrt(n) features: 24.9%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 10:
# random objects, n features      : 99.4%
# random objects, sqrt(n) features: 46.5%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 11:
# random objects, n features      : 99.6%
# random objects, sqrt(n) features: 79.7%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 12:
# random objects, n features      : 89.1%
# random objects, sqrt(n) features: 41.5%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 13:
# random objects, n features      : 70.0%
# random objects, sqrt(n) features: 28.5%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 14:
# random objects, n features      : 96.8%
# random objects, sqrt(n) features: 34.3%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 15:
# random objects, n features      : 99.8%
# random objects, sqrt(n) features: 95.1%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 16:
# random objects, n features      : 99.1%
# random objects, sqrt(n) features: 48.4%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 17:
# random objects, n features      : 83.4%
# random objects, sqrt(n) features: 22.4%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 18:
# random objects, n features      : 93.2%
# random objects, sqrt(n) features: 54.7%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 19:
# random objects, n features      : 85.0%
# random objects, sqrt(n) features: 27.0%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 20:
# random objects, n features      : 97.1%
# random objects, sqrt(n) features: 72.1%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset 21:
# random objects, n features      : 81.7%
# random objects, sqrt(n) features: 29.9%
