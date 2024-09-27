from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
import random


def normalization(data):
    data_new = [nums.copy() for nums in data]
    for j in range(len(data[0])):
        min_val = np.min(data[:, j])
        max_val = np.max(data[:, j])
        for i in range(len(data)):
            data_new[i][j] = (data[i][j] - min_val)/(max_val - min_val)

    return data_new

def k_means(data, num_clusters):
    # Инициализация начальных точек
    mean_points = [np.zeros(len(numX[0])) for _ in range(num_clusters)]

    # Рассчитываем начальные значения для центров кластеров
    for point in mean_points:
        for j in range(len(numX[0])):
            min_val = np.min(numX[:, j])
            max_val = np.max(numX[:, j])
            point[j] = random.uniform(min_val, max_val)

    prev_points = []
    while not prev_points or not np.array_equal(prev_points, mean_points):
        # Копируем значения, а не ссылки
        prev_points = [point.copy() for point in mean_points]

        # Создание пустых кластеров
        clusters = [[] for _ in range(num_clusters)]

        # Присваиваем точки к кластерам
        for i in range(len(numX)):
            min_linalg_norm = float('inf')
            num_cluster = 0
            for j in range(len(mean_points)):
                linalg_norm = np.linalg.norm(mean_points[j] - numX[i])
                if linalg_norm < min_linalg_norm:
                    min_linalg_norm = linalg_norm
                    num_cluster = j

            clusters[num_cluster].append(i)

        # Пересчитываем центры кластеров
        mean_points = [np.zeros(len(numX[0])) for _ in range(num_clusters)]

        for j in range(len(clusters)):
            if len(clusters[j]) == 0:
                # Если кластер пустой, оставляем старое значение
                mean_points[j] = prev_points[j]
                continue

            # Рассчитываем новое среднее значение для каждого кластера
            for k in range(len(clusters[j])):
                mean_points[j] += numX[clusters[j][k]]
            mean_points[j] *= (1. / len(clusters[j]))

    return clusters


# загрузка набора данных
wine = fetch_ucirepo(id=109)

# данные в формате pandas dataframes
X = wine.data.features
y = wine.data.targets

numX = wine.data.features.to_numpy()

num_clusters = 3
# Вывод кластеров
# Запись в файл y
with open('y.txt', 'w') as f:
        f.write(str(y))  # Печатает каждое значение с новой строки

# Запись в файл clusters
with open('clusters.txt', 'w') as f:
    for cluster in clusters:
        f.write(f"{cluster}\n")  # Печатает каждый кластер в виде списка


clusters_vis = np.zeros(len(numX), dtype=int)
for cluster_id, cluster in enumerate(clusters):
    for i in cluster:
        clusters_vis[i] = cluster_id

mean_points_np = np.array(mean_points)  # Преобразуем список в numpy-массив
plt.scatter(numX[:, 0], numX[:, 12], c=clusters_vis, cmap='viridis') # оси -- первые два столбца
plt.scatter(mean_points_np[:, 0], mean_points_np[:, 12], c='black', s=200, alpha=0.5)
plt.show()
