import numpy as np
import knn_functions as knn
import matplotlib.pyplot as plt

# Reading dataset, changing features name and reorganizing columns
filename = 'tae.csv'
dataframe = knn.read_dataset(filename)
# Applying one-hot encoding for categorical features
dataframe_encoded = knn.one_hot(dataframe)
# Applying min-max normalization
minmax = knn.min_max(dataframe_encoded.to_numpy())
knn.normalize(dataframe_encoded, minmax)

window_types = ["fixed", "variable"]
kernel_functions_list = knn.kernel_functions.keys()
distance_functions_list = knn.distance_functions.keys()
result = list()

for k_f in kernel_functions_list:
    print(k_f)
    for d_f in distance_functions_list:
        for w_t in window_types:
            knn.kernel_f_type = k_f
            knn.distance_type = d_f
            if w_t == "fixed":
                for h in np.linspace(0, 20, 101):
                    result.append([w_t, h, k_f, d_f, knn.count_f_score(dataframe, 3, h)])
            else:
                for k in range(1, 25):
                    result.append([w_t, k, k_f, d_f, knn.count_f_score(dataframe, 3, 0, k)])

result.sort(key=lambda z: z[4], reverse=True)
best = result[0]
knn.kernel_f_type = best[2]
knn.distance_type = best[3]

# Saved data (best result)
# best = ['fixed', 1.2000000000000002, 'quartic', 'manhattan', 0.6510418016990313]
# knn.kernel_f_type = "quartic"
# knn.distance_type = "manhattan"

x_h = list()
x_k = list()
y_h = list()
y_k = list()

for h in np.linspace(0, 20, 101):
    x_h.append(h)
    y_h.append(knn.count_f_score(dataframe, 3, h))

# Plot for F-score / window width
plt.plot(x_h, y_h)
plt.xlabel('Window width')
plt.ylabel('F-score:macro')
plt.show()

for k in range(1, 25):
    x_k.append(k)
    y_k.append(knn.count_f_score(dataframe, 3, 0, k))

# Plot for F-score / neighbours count
plt.plot(x_k, y_k)
plt.xlabel('Neighbours')
plt.ylabel('F-score:macro')
plt.show()
