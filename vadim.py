from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets

numX = wine.data.features.to_numpy()

num_clusters = 2

mean_points  = np.array([[0.]*len(numX[0])])*num_clusters

for point in mean_points:
    for j in range(len(numX[0])):
        min_val = numX[0][j]
        max_val = numX[0][j]
        for i in range(len(numX)):
            min_val = min_val if min_val < numX[i][j] else numX[i][j]
            max_val = max_val if max_val > numX[i][j] else numX[i][j]

        point[j] = min_val + (j+1)/(len(mean_points) + 2) * (max_val - min_val)

prev_points = []
while not prev_points or prev_points != mean_points:
    prev_points = mean_points
    clusters = []*num_clusters
    for i in range(len(numX)):
        min_linalg_norm = 0.
        num_cluster = 0
        for j in range(len(mean_points)):
            linalg_norm = np.linalg.norm(mean_points[j] - numX[i])
            if linalg_norm < min_linalg_norm:
                min_linalg_norm = linalg_norm
                num_cluster = j

        print(num_cluster)
        clusters[num_cluster].append(i)





print(numX)