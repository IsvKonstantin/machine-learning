import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.utils import shuffle


def read_data():
    data_chips = pd.read_csv('chips.csv')
    data_geyser = pd.read_csv('geyser.csv')
    data_chips['class'] = data_chips['class'].apply(lambda x: 1 if x == 'P' else -1)
    data_geyser['class'] = data_geyser['class'].apply(lambda x: 1 if x == 'P' else -1)
    # data_chips = shuffle(data_chips)
    # data_geyser = shuffle(data_geyser)

    return data_chips, data_geyser


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.classifiers = list()
        self.classifier_weights = list()

    @staticmethod
    def __calculate_error(weights, data_x, data_y, prediction):
        return sum([weights[i] for i in range(data_x.shape[0]) if prediction[i] != data_y[i]])

    def fit(self, data_x, data_y):
        sample_weights = np.ones(data_x.shape[0]) / data_x.shape[0]
        for i in range(self.n_estimators):
            classifier = DecisionTreeClassifier(max_depth=1)
            self.classifiers.append(classifier)

        for i in range(self.n_estimators):
            classifier = self.classifiers[i]
            classifier.fit(data_x, data_y, sample_weight=sample_weights)
            prediction = classifier.predict(data_x)

            error = self.__calculate_error(sample_weights, data_x, data_y, prediction)
            alpha = 0.5 * np.log((1 - error) / error)
            sample_weights *= np.exp(-alpha * prediction * data_y)
            sample_weights /= np.sum(sample_weights)

            self.classifier_weights.append(alpha)

    def predict(self, data_x):
        predictions = np.zeros(data_x.shape[0])
        for alpha, clf in zip(self.classifier_weights, self.classifiers):
            predictions += alpha * clf.predict(data_x)
        return np.sign(predictions)


def accuracy_score(actual, predicted):
    score = sum([int(a == p) for a, p in zip(actual, predicted)]) / len(actual)
    return score


def draw_accuracy_graph(data, n, data_name):
    data_x = np.delete(data.to_numpy(), 2, axis=1)
    data_y = data.to_numpy().T[2]
    plot_x, plot_y = list(), list()

    for i in range(1, n):
        classifier = AdaBoost(i)
        classifier.fit(data_x, data_y)
        plot_x.append(i)
        plot_y.append(accuracy_score(data_y, classifier.predict(data_x)))

    plt.plot(plot_x, plot_y)
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy score')
    plt.title(data_name)
    plt.show()


def draw_meshgrid(data, data_name, n_estimators):
    data_x = np.delete(data.to_numpy(), 2, axis=1)
    data_y = data.to_numpy().T[2]
    clf = AdaBoost(n_estimators)
    clf.fit(data_x, data_y)

    min_x, min_y = np.amin(data_x, axis=0) - [0.1, 0.1]
    max_x, max_y = np.amax(data_x, axis=0) + [0.1, 0.1]
    x, y = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
    predictions = clf.predict(np.column_stack([x.reshape((-1)), y.reshape((-1))]))
    n_x, n_y = data_x[data_y == -1].T
    p_x, p_y = data_x[data_y == 1].T

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.pcolormesh(x, y, predictions.reshape(x.shape), cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), shading='gouraud')
    plt.scatter(n_x, n_y, marker='_', color='red', s=45)
    plt.scatter(p_x, p_y, marker='+', color='blue', s=45)
    plt.title('{}, {:.3}%'.format(data_name, 100 * accuracy_score(data_y, clf.predict(data_x))))
    plt.show()


def process_dataset(data, data_name):
    draw_accuracy_graph(data, 100, data_name)

    boosting_steps = [1, 2, 5, 8, 13, 34, 55, 100, 200, 500, 1000]
    for step in boosting_steps:
        draw_meshgrid(data, data_name, step)


data_chips, data_geyser = read_data()
process_dataset(data_chips, 'Chips')
# process_dataset(data_geyser, 'Geyser')
