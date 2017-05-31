import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('bmh')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[8, 7], [6, 9], [9, 4]]}
new_features = [5, 7]


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is greater than classes number!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

new_features_group = k_nearest_neighbors(dataset, new_features)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], color=i)

plt.scatter(new_features[0], new_features[1], color=new_features_group)
plt.show()