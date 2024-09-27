from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets

numX = wine.data.features.to_numpy()

num_clusters = 2

mean_points  = np.array([0.]*num_clusters)



print(numX)