from os import listdir
from os.path import isfile, join
from math import log
import matplotlib.pyplot as plt

def read_data(index, n):
    path = 'data/part{}/'.format(index)
    files = [file for file in listdir(path) if isfile(join(path, file))]
    data_x, data_y = list(), list()
    for file in files:
        with open(join(path, file)) as opened_file:
            lines = map(lambda l: l[:-1], opened_file.readlines())
            lines = list(filter(lambda l: l != '', map(lambda l: l.replace('Subject: ', ''), lines)))

        temp_x = list()
        for line in lines:
            temp_x += [number for number in line.split(' ') if number != '']

        temp_x = list(map(lambda x: str.join(" ", [temp_x[j] for j in range(x, x + n)]), range(len(temp_x) - n)))
        data_x.append(temp_x)
        data_y.append('legit' if 'legit' in file else 'spam')

    return data_x, data_y


class NaiveBayes:
    def __init__(self, alpha=1.0, lambda_legit=1.0, lambda_spam=1.0):
        self.alpha = alpha
        self.lambda_legit = lambda_legit
        self.lambda_spam = lambda_spam
        self.spam_count = 0
        self.legit_count = 0
        self.frequencies = {}

    def fit(self, data_x, data_y):
        self.spam_count = 0
        self.legit_count = 0
        self.frequencies, frequencies = {}, {}
        for x, y in zip(data_x, data_y):
            for word in x:
                if frequencies.get((y, word), 0) != 0:
                    frequencies[y, word] += 1
                else:
                    frequencies[y, word] = 1

            if y == 'legit':
                self.legit_count += 1
            else:
                self.spam_count += 1

        self.frequencies = frequencies

    def make_prediction(self, data_x):
        probability_legit = log(self.lambda_legit * self.legit_count / (self.legit_count + self.spam_count))
        probability_spam = log(self.lambda_spam * self.spam_count / (self.legit_count + self.spam_count))
        a = self.alpha
        for word in data_x:
            probability_legit += log((self.frequencies.get(('legit', word), 0) + a) / (self.legit_count + 2 * a))
            probability_spam += log((self.frequencies.get(('spam', word), 0) + a) / (self.spam_count + 2 * a))

        return 'legit' if probability_legit > probability_spam else 'spam', probability_legit


def merge_data(data):
    data_x, data_y = list(), list()
    for d in data:
        data_x += d[0]
        data_y += d[1]

    return data_x, data_y


def accuracy_score(actual, predicted):
    score = sum([int(a == p) for a, p in zip(actual, predicted)]) / len(actual)
    return score


def calculate_FN(actual, predicted):
    count = 0
    for a, p in zip(actual, predicted):
        if a == 'legit' and p == 'spam':
            count += 1
    return count


def cross_validation(data, folds, model):
    accuracy = 0
    false_negative_count = 0

    for i in range(folds):
        data_test = data[i]
        data_train = merge_data(data[:i] + data[i + 1:])
        model.fit(data_train[0], data_train[1])

        actual = data_test[1]
        predicted = [model.make_prediction(x)[0] for x in data_test[0]]
        false_negative_count += calculate_FN(actual, predicted)
        accuracy += accuracy_score(actual, predicted)

    return accuracy / folds, false_negative_count == 0


def find_best_alpha(data):
    """
    Alpha = 10,     accuracy = 0.771
    Alpha = 5,      accuracy = 0.811
    Alpha = 2,      accuracy = 0.85
    Alpha = 1,      accuracy = 0.867
    Alpha = 0.5,    accuracy = 0.886
    Alpha = 0.1,    accuracy = 0.92
    Alpha = 0.1,    accuracy = 0.92
    Alpha = 0.01,   accuracy = 0.95
    Alpha = 0.001,  accuracy = 0.958
    Alpha = 0.0001, accuracy = 0.967
    Alpha = 1e-05,  accuracy = 0.972
    Alpha = 1e-06,  accuracy = 0.976
    Alpha = 1e-07,  accuracy = 0.975
    Alpha = 1e-08,  accuracy = 0.976
    Alpha = 1e-09,  accuracy = 0.977 <--- best
    Alpha = 1e-10,  accuracy = 0.976
    Alpha = 1e-15,  accuracy = 0.977
    Alpha = 1e-20,  accuracy = 0.976
    Alpha = 1e-30,  accuracy = 0.974
    """
    alphas = [10, 5, 2, 1, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-15, 1e-20, 1e-30]
    best_accuracy = 0
    best_alpha = alphas[0]
    for alpha in alphas:
        model = NaiveBayes(alpha=alpha)
        accuracy, _ = cross_validation(data, 10, model)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
    return best_alpha


def calculate_TPR(actual, predicted):
    true_positive = 0
    false_negative = 0

    for a, p in zip(actual, predicted):
        if a == 'legit' and p == 'legit':
            true_positive += 1
        if a == 'legit' and p == 'spam':
            false_negative += 1

    return true_positive / (true_positive + false_negative)


def calculate_FPR(actual, predicted):
    false_positive = 0
    true_negative = 0

    for a, p in zip(actual, predicted):
        if a == 'spam' and p == 'legit':
            false_positive += 1
        if a == 'spam' and p == 'spam':
            true_negative += 1

    return false_positive / (false_positive + true_negative)


def find_lambda_legit(data, alpha):
    lambdas = [10 ** (20 * p) for p in range(31)]
    plot_x, plot_y = list(), list()

    for lambda_ in lambdas:
        model = NaiveBayes(alpha=alpha, lambda_legit=lambda_)
        accuracy, false_negative_count = cross_validation(data, 10, model)
        plot_x.append(log(lambda_, 10))
        plot_y.append(accuracy)
        if false_negative_count:
            plt.plot(plot_x, plot_y)
            plt.xlabel('Lambda legit, degree')
            plt.ylabel('Accuracy')
            plt.show()
            return lambda_
    return -1


def ROC_curve(data, model):
    data_x, data_y = merge_data(data)
    model.fit(data_x, data_y)
    results = list()

    for x, y in zip(data_x, data_y):
        results.append([y, model.make_prediction(x)[1]])

    results.sort(key=lambda z: z[1], reverse=True)
    legit_count = 0
    spam_count = 0
    for y in data_y:
        if y == 'legit':
            legit_count += 1
        else:
            spam_count += 1

    x_step = 1 / spam_count
    y_step = 1 / legit_count
    current_x, current_y = 0, 0
    plot_x, plot_y = list(), list()

    for r in results:
        if r[0] == 'legit':
            current_y += y_step
        else:
            current_x += x_step
        plot_x.append(current_x)
        plot_y.append(current_y)

    plt.plot(plot_x, plot_y)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()



data = [read_data(i + 1, 1) for i in range(10)]
# alpha = find_best_alpha(data)                 # 1e-9
# lambda_ = find_lambda_legit(data, alpha)      # 1e300
# model = NaiveBayes(alpha=alpha, lambda_legit=lambda_)
find_lambda_legit(data, 1e-9)
model = NaiveBayes(alpha=1e-9, lambda_legit=1e300)
ROC_curve(data, model)
