import numpy as np
import math as m
import random
import matplotlib.pyplot as plt

def read_data(filename):
    features = 0
    n_train = 0
    n_test = 0
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                features = int(line)
            elif i == 1:
                n_train = int(line)
            elif n_train != 0 and i == n_train + 2:
                n_test = int(line)
                break

    train_data = np.loadtxt(filename, dtype=np.float64, skiprows=2, max_rows=n_train)
    test_data = np.loadtxt(filename, dtype=np.float64, skiprows=2 + n_train + 1, max_rows=n_test)
    return train_data, test_data, features


def remove_redundant(train_data, test_data):
    data = np.concatenate((train_data, test_data))
    indexes = list()

    for i, column in enumerate(data.T):
        if i != test_data.shape[1] - 1 and np.min(column) == np.max(column):
            indexes.append(i)

    return np.delete(train_data, indexes, axis=1), np.delete(test_data, indexes, axis=1)


#
def normalize(dataframe, minmax):
    # Normalizing dataframe values.
    # Last column "Labels" is skipped.
    for i, feature in enumerate(list(dataframe)[:-1]):
        dataframe[feature] = dataframe[feature].apply(lambda x: (x - minmax[i][0]) / (minmax[i][1] - minmax[i][0]))


def min_max_normalization(data):
    min_max = list()
    for column in data.T[:-1]:
        min_max.append([np.min(column), np.max(column)])

    for i, row in enumerate(data):
        for j in range(len(row) - 1):
            data[i][j] = (data[i][j] - min_max[j][0]) / (min_max[j][1] - min_max[j][0])

    return data


def z_score_normalization(data):
    mean_std = list()
    for column in data.T[:-1]:
        mean_std.append([np.mean(column), np.std(column)])

    for j, column in enumerate(data.T[:-1]):
        for i in range(len(column)):
            data[i][j] = (data[i][j] - mean_std[j][0]) / mean_std[j][1]

    return data


def calculate_nrmse(actual, predicted):
    rmse = m.sqrt(sum([(a - p) ** 2 for a, p in zip(actual, predicted)]) / len(actual))
    nrmse = rmse / (max(actual) - min(actual))
    return nrmse


def least_squares(data, t):
    features = data.shape[1] - 1
    data_x = np.delete(data, features, axis=1)
    data_y = data.T[features]
    V, D, UT = np.linalg.svd(data_x, full_matrices=False)
    D = np.diag(D)
    # U * (D^2 + t * In)^-1 * D * V.T * y

    # D = np.linalg.inv(np.diag(D))
    # F_pinv = UT.T.dot(D.dot(V.T))
    # weights = F_pinv.dot(data_y)

    weights = (UT.T.dot(np.linalg.inv(D.dot(D) + t * np.diag(np.ones(D.shape[0]))))).dot(D).dot(V.T).dot(data_y)
    return weights


def get_weights(n):
    return [random.uniform(-1 / (2 * n), 1 / (2 * n)) for i in range(n)]


def gradient_descent(train_data, test_data, steps, t):
    features = train_data.shape[1] - 1

    train_data_x = np.delete(train_data, features, axis=1)
    test_data_x = np.delete(test_data, features, axis=1)

    train_data_y = train_data.T[features]
    test_data_y = test_data.T[features]

    weights = np.array(get_weights(features))
    errors = list()
    alpha = 1e-16

    for step in range(steps):
        i = np.random.randint(0, len(train_data_x))
        gradient = (np.dot(train_data_x[i], weights) - train_data_y[i]) * train_data_x[i]

        weights = weights * (1 - t * alpha) - alpha * gradient

        if step % 10 == 0:
            error_train = calculate_nrmse(train_data_y, train_data_x.dot(weights))
            error_test = calculate_nrmse(test_data_y, test_data_x.dot(weights))
            errors.append([step, error_train, error_test])

    return weights, errors


def leave_one_out(data, method, t):
    error = 0
    fold_size = 4

    for i in range(0, len(data), fold_size):
        test_data = data[i:i + fold_size]
        train_data = np.concatenate((data[0:i], data[i + fold_size:data.shape[0]]), axis=0)
        features = train_data.shape[1] - 1

        test_data_x = np.delete(test_data, features, axis=1)
        test_data_y = test_data.T[features]

        if method == 'gradient_descent':
            weights, _ = gradient_descent(train_data, test_data, 1000, t)
            error += calculate_nrmse(test_data_y, np.dot(test_data_x, weights))
        else:  # 'least_squares'
            weights = least_squares(train_data, t)
            error += calculate_nrmse(test_data_y, np.dot(test_data_x, weights))

    return error / len(data)


def find_regression_parameter(data, method, values):
    algorithms = list()
    for i in values:  # 0 60 4
        print(i)
        algorithms.append([i, leave_one_out(data, method, i)])

    algorithms.sort(key=lambda a: a[1])
    return algorithms


def generate_new_weights(weights, i):
    return np.array([w + np.random.randint(-50, 50) for w in weights])


def calculate_energy(y, x, w):
    return calculate_nrmse(y, np.dot(x, w))


def check_transition(delta, temperature):
    if np.random.random(1)[0] < np.exp(-delta / temperature):
        return True
    else:
        return False


def get_temperature(initial, i):
    return initial / i


def simulated_annealing(train_data, test_data):
    features = train_data.shape[1] - 1

    train_data_x = np.delete(train_data, features, axis=1)
    test_data_x = np.delete(test_data, features, axis=1)

    train_data_y = train_data.T[features]
    test_data_y = test_data.T[features]

    current_weights = np.array(get_weights(features))
    temperature = 1000
    min_temperature = 0.1
    current_energy = calculate_energy(train_data_y, train_data_x, current_weights)
    errors = list()

    for i in range(1, 1000):
        new_weights = generate_new_weights(current_weights, i)
        new_energy = calculate_energy(train_data_y, train_data_x, new_weights)

        if new_energy < current_energy:
            current_energy = new_energy
            current_weights = new_weights
            print(i)
        else:
            if check_transition(new_energy - current_energy, temperature):
                current_energy = new_energy
                current_weights = new_weights

        if i % 10 == 0:
            error_train = calculate_nrmse(train_data_y, train_data_x.dot(current_weights))
            error_test = calculate_nrmse(test_data_y, test_data_x.dot(current_weights))
            errors.append([i, error_train, error_test])

        temperature *= 0.99
        if temperature < min_temperature:
            break

    return current_weights, errors


train_data, test_data, _ = read_data('5.txt')
np.random.shuffle(train_data)
np.random.shuffle(test_data)
train_data, test_data = remove_redundant(train_data, test_data)

# train_data = min_max_normalization(train_data)
# test_data = min_max_normalization(test_data)
# train_data = z_score_normalization(train_data)
# test_data = z_score_normalization(test_data)

train_data = np.insert(train_data, 0, np.ones(train_data.shape[0]), axis=1)
test_data = np.insert(test_data, 0, np.ones(test_data.shape[0]), axis=1)

features = train_data.shape[1] - 1
train_data_x = np.delete(train_data, features, axis=1)
test_data_x = np.delete(test_data, features, axis=1)

train_data_y = train_data.T[features]
test_data_y = test_data.T[features]

iterations = 1000

# algorithms_least_squares = find_regression_parameter(train_data, 'least_squares', np.linspace(0, 0.00001, 15))
# algorithms_gradient_descent = find_regression_parameter(train_data, 'gradient_descent', np.linspace(0, 0.01, 15))

weights = least_squares(train_data, 100)
error_test = calculate_nrmse(test_data_y, np.dot(test_data_x, weights))
error_train = calculate_nrmse(train_data_y, np.dot(train_data_x, weights))

weights, errors = gradient_descent(train_data, test_data, iterations, 0)
x_axis = list(map(lambda x: x[0], errors))
errors_train = list(map(lambda x: x[1], errors))
errors_test = list(map(lambda x: x[2], errors))

plt.plot(list(range(0, iterations)), [error_train] * iterations, label='squares test')
plt.plot(list(range(0, iterations)), [error_test] * iterations, label='squares test')
plt.plot(x_axis, errors_train, label='grad train')
plt.plot(x_axis, errors_test, label='grad test')


_, errors = simulated_annealing(train_data, test_data)
x_axis = list(map(lambda x: x[0], errors))
errors_train = list(map(lambda x: x[1], errors))
errors_test = list(map(lambda x: x[2], errors))
plt.plot(x_axis, errors_train, label='ann train')
plt.plot(x_axis, errors_test, label='ann test')

plt.legend()
plt.show()
