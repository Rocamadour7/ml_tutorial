import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')

X = np.array([[1, 1], [2, 3], [1, 3], [7, 8], [6, 9], [6, 7]])

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['g.', 'm.', 'c.', 'b.', 'k.', 'w.', 'r.', 'y.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x')
plt.show()