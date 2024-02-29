import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euc_dist(data_points, centroids):
        return np.sqrt(np.sum((centroids - data_points) ** 2, axis=1))

    def fit(self, x, max_iterations=200):
        global y
        self.centroids = np.random.uniform(np.amin(x, axis=0), np.amax(x, axis=0), size=(self.k, x.shape[1]))

        for _ in range(max_iterations):
            y = []
            for data_point in x:
                distances = KMeansClustering.euc_dist(data_point, self.centroids)
                cluster_number = np.argmin(distances)
                y.append(cluster_number)

            y = np.array(y)
            cluster_index = []
            for i in range(self.k):
                cluster_index.append(np.argwhere(y == i))
            cluster_centers = []

            for i, indices in enumerate(cluster_index):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])

                else:
                    cluster_centers.append(np.mean(x[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break

            else:
                self.centroids = np.array(cluster_centers)

        return y


data = make_blobs(n_samples=100, n_features=2, centers=3)
random_points = data[0]
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)

ari = adjusted_rand_score(data[1], labels)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker="*", s=200)
print(ari)
plt.show()
