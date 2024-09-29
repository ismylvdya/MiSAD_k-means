import numpy as np
import matplotlib.pyplot as plt
import random
from ucimlrepo import fetch_ucirepo

def normalization(data):
    data_new = np.copy(data)
    for j in range(len(data[0])):
        min_val = np.min(data[:, j])
        max_val = np.max(data[:, j])
        for i in range(len(data)):
            data_new[i][j] = (data[i][j] - min_val) / (max_val - min_val)
    return data_new

def k_means(data, num_clusters):
    mean_points = [np.zeros(len(data[0])) for _ in range(num_clusters)]

    for point in mean_points:
        for j in range(len(data[0])):
            min_val = np.min(data[:, j])
            max_val = np.max(data[:, j])
            point[j] = random.uniform(min_val, max_val)

    prev_points = []
    while not prev_points or not np.array_equal(prev_points, mean_points):
        prev_points = [point.copy() for point in mean_points]

        clusters = [[] for _ in range(num_clusters)]

        for i in range(len(data)):
            min_linalg_norm = float('inf')
            num_cluster = 0
            for j in range(len(mean_points)):
                linalg_norm = np.linalg.norm(mean_points[j] - data[i])
                if linalg_norm < min_linalg_norm:
                    min_linalg_norm = linalg_norm
                    num_cluster = j

            clusters[num_cluster].append(i)

        mean_points = [np.zeros(len(data[0])) for _ in range(num_clusters)]

        for j in range(len(clusters)):
            if len(clusters[j]) == 0:
                mean_points[j] = prev_points[j]
                continue

            for k in range(len(clusters[j])):
                mean_points[j] += data[clusters[j][k]]
            mean_points[j] *= (1. / len(clusters[j]))

    return [clusters, mean_points]

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

numX = wine.data.features.to_numpy()

num_clusters = 3

data_new = normalization(numX)
clusters_finish = k_means(data_new, num_clusters)

# Запись результатов в файлы
with open('y.txt', 'w') as f:
    f.write(str(y))

with open('clusters.txt', 'w') as f:
    for cluster in clusters_finish[0]:
        f.write(f"{cluster}\n")

clusters_vis = np.zeros(len(numX), dtype=int)
for cluster_id, cluster in enumerate(clusters_finish[0]):
    for i in cluster:
        clusters_vis[i] = cluster_id

mean_points_np = np.array(clusters_finish[1])

plt.scatter(data_new[:, 12], data_new[:, 2], c=clusters_vis, cmap='viridis')
plt.scatter(mean_points_np[:, 12], mean_points_np[:, 2], c='black', s=200, alpha=0.5)

plt.show()