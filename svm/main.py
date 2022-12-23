import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def read_data():
    data_chips = pd.read_csv('chips.csv')
    data_geyser = pd.read_csv('geyser.csv')
    data_chips['class'] = data_chips['class'].apply(lambda x: 1 if x == 'N' else -1)
    data_geyser['class'] = data_geyser['class'].apply(lambda x: 1 if x == 'N' else -1)
    data_chips = shuffle(data_chips)
    data_geyser = shuffle(data_geyser)

    return data_chips, data_geyser


# noinspection PyPep8Naming
class SVM:
    def __init__(self, iterations=200, kernel_f_type='linear', c=50.0, eps=10e-11, p=3.0):
        self.kernel_f_type = kernel_f_type
        self.iterations = iterations
        self.kernel = self.set_kernel(kernel_f_type, p)
        self.K = None
        self.c = c
        self.eps = eps
        self.data_x = None
        self.data_y = None
        self.a = None
        self.b = None

    def set_kernel(self, kernel_f_type='linear', p=3.0):
        kernel_functions = {
            "linear": lambda x, y: np.dot(x, y),
            "gaussian": lambda x, y: np.exp(-1 * p * np.sum((x - y) ** 2)),
            "polynomial": lambda x, y: (np.dot(x, y)) ** p,
        }
        self.kernel = kernel_functions[kernel_f_type]
        return self.kernel

    def __calculate_kernel_matrix(self):
        size = self.data_x.shape[0]
        k = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                k[i][j] = self.kernel(self.data_x[i], self.data_x[j])
        return k

    def make_prediction(self, test_data_x, data_x=None, data_y=None, a=None, b=None):
        if data_x is None and data_y is None and a is None and b is None:
            data_x = self.data_x
            data_y = self.data_y
            a = self.a
            b = self.b

        kernel_matrix = np.array([self.kernel(test_data_x, data_x[i]) for i in range(data_x.shape[0])])
        predicted = b + np.dot(a * data_y, kernel_matrix)
        return 1 if predicted >= 0 else -1

    def fit(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.K = self.__calculate_kernel_matrix()
        kernel_matrix = self.K
        n = self.data_x.shape[0]
        a = np.zeros(n)
        y = self.data_y
        c = self.c
        b = 0.0
        e = self.eps

        for step in range(self.iterations):
            for i in range(n):
                j = random.choice(list(range(0, i)) + list(range(i + 1, n)))
                e_i = b + np.dot(a * y, kernel_matrix.T[i]) - y[i]
                e_j = b + np.dot(a * y, kernel_matrix.T[j]) - y[j]
                a_i = a[i]
                a_j = a[j]
                L, H = 0.0, 0.0

                if y[i] == y[j]:
                    L = max(a[i] + a[j] - c, 0.0)
                    H = min(a[i] + a[j], c)
                else:
                    L = max(a[j] - a[i], 0.0)
                    H = min(c + a[j] - a[i], c)

                if abs(H - L) < e:
                    continue

                n_j = 2.0 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j]
                if n_j >= 0:
                    continue

                a[j] -= y[j] * (e_i - e_j) / n_j
                a[j] = max(min(a[j], H), L)
                if abs(a[j] - a_j) < e:
                    continue

                a[i] += y[i] * y[j] * (a_j - a[j])
                b1 = b - e_i - y[i] * (a[i] - a_i) * kernel_matrix[i][i] - y[j] * (a[j] - a_j) * kernel_matrix[i][j]
                b2 = b - e_j - y[i] * (a[i] - a_i) * kernel_matrix[i][j] - y[j] * (a[j] - a_j) * kernel_matrix[j][j]
                b = b1 if 0 < a[i] < c else (b2 if 0 < a[j] < c else (b1 + b2) / 2)

        self.a, self.b = a, b
        return a, b


def accuracy_score(predicted, actual):
    score = 0
    for a, p in zip(actual, predicted):
        if a == p:
            score += 1
    return score / len(actual)


def cross_validation(data, model):
    data = data.to_numpy()
    folds = 10
    split = np.array_split(data, folds)
    average_accuracy = 0

    for i in range(len(split)):
        data_train = np.concatenate(split[:i] + split[i + 1:])
        data_test = split[i]
        data_train_x, data_test_x = np.delete(data_train, 2, axis=1), np.delete(data_test, 2, axis=1)
        data_train_y, data_test_y = data_train.T[2], data_test.T[2]

        predicted = np.array([model.make_prediction(x) for x in data_test_x])
        actual = data_test_y
        average_accuracy += accuracy_score(predicted, actual)

    average_accuracy /= folds
    return average_accuracy


def find_best_parameters(data, kernel_f_type):
    if kernel_f_type == 'gaussian':
        kernel_params = np.linspace(2, 5, num=7)
    else:
        kernel_params = [2, 3, 4, 5]

    C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    best_parameters = list()
    best_accuracy = 0

    for c in C:
        print('Testing C')
        for p in kernel_params:
            model = SVM(kernel_f_type=kernel_f_type, c=c, p=p)
            average_accuracy = cross_validation(data, model)

            if average_accuracy > best_accuracy:
                best_accuracy = average_accuracy
                best_parameters = [c, p, average_accuracy]

    return best_parameters


def draw(data, data_name, model):
    data_x = np.delete(data.to_numpy(), 2, axis=1)
    data_y = data.to_numpy().T[2]
    model.fit(data_x, data_y)

    x0_N = list()
    x1_N = list()
    x0_P = list()
    x1_P = list()
    for i in np.linspace(np.min(data_x.T[0]), np.max(data_x.T[0]), 100):
        for j in np.linspace(np.min(data_x.T[1]), np.max(data_x.T[1]), 80):
            if model.make_prediction(np.array([i, j])) == 1:
                x0_N.append(i)
                x1_N.append(j)
            else:
                x0_P.append(i)
                x1_P.append(j)

    predicted = list()
    for x in data_x:
        predicted.append(model.make_prediction(x))

    accuracy = accuracy_score(np.array([model.make_prediction(x) for x in data_x]), data_y) * 100
    plt.title('{}: kernel = {}, accuracy = {:.3}%'.format(data_name, model.kernel_f_type, accuracy))

    plt.scatter(x0_N, x1_N, alpha=0.1, c='purple')
    plt.scatter(x0_P, x1_P, alpha=0.1, c='red')

    x0 = [x[0] for i, x in enumerate(data_x) if data_y[i] == 1]
    x1 = [x[1] for i, x in enumerate(data_x) if data_y[i] == 1]
    plt.scatter(x0, x1, c='purple')
    x0 = [x[0] for i, x in enumerate(data_x) if data_y[i] != 1]
    x1 = [x[1] for i, x in enumerate(data_x) if data_y[i] != 1]
    plt.scatter(x0, x1, c='red')
    plt.show()


data_chips, data_geyser = read_data()

# best_parameters_chips = {
#     'linear': find_best_parameters(data_chips, 'linear'),           # 100 --- 2 --- 53%
#     'gaussian': find_best_parameters(data_chips, 'gaussian'),       # 5   --- 4 --- 83%
#     'polynomial': find_best_parameters(data_chips, 'polynomial')    # 100 --- 4 --- 75%
# }

# best_parameters_geyser = {
#     'linear': find_best_parameters(data_geyser, 'linear'),          # 0.5 --- 2 --- 90%
#     'gaussian': find_best_parameters(data_geyser, 'gaussian'),      # 0.5 --- 2 --- 88%
#     'polynomial': find_best_parameters(data_geyser, 'polynomial')   # 0.05 -- 2 --- 87%
# }

draw(data_chips, 'Chips', SVM(kernel_f_type='linear', c=100, p=2))
draw(data_chips, 'Chips', SVM(kernel_f_type='gaussian', c=5, p=4))
draw(data_chips, 'Chips', SVM(kernel_f_type='polynomial', c=100, p=4))

draw(data_geyser, 'Geyser', SVM(kernel_f_type='linear', c=0.5, p=2))
draw(data_geyser, 'Geyser', SVM(kernel_f_type='gaussian', c=0.5, p=2))
draw(data_geyser, 'Geyser', SVM(kernel_f_type='polynomial', c=0.05, p=2))
