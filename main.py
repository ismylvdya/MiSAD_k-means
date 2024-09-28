import numpy as np
import matplotlib.pyplot as plt

def normalized(data):
    '''Функция для нормализации данных
    принимает на вход двумерный numpy массив
    возвращает нормализованный массив (в каждом столбце теперь значения из [0,1])'''

    data_new = np.copy(data)
    for j in range(len(data[0])): # перебираем все строки датасета
        min_val = np.min(data[:, j])
        max_val = np.max(data[:, j])
        for i in range(len(data)):
            data_new[i][j] = (data[i][j] - min_val) / (max_val - min_val)
    return data_new

def distance(x1, x2):
    '''Функция для вычисления расстояния между двумя точками'''
    return np.sqrt(np.sum((x1 - x2)**2))

def initialize_centers(X, k):
    '''Функция для инициализации центров кластеров'''
    centers = np.zeros((k, X.shape[1])) # двумерный массив (количество центроидов x координаты каждого). X.shape[1] -- количество столбцов в датасете (количество осей)
    for i in range(k):
        centers[i] = X[np.random.choice(range(X.shape[0]))] # каждый из k кластеров инициализируем координатами случайной точки из датасета
    return centers

def assign_clusters(X, centers):
    '''Функция для разбиения данных на кластеры'''
    clusters = [] # массив распределения точек по кластерам. Он длины количества точек и каждый его элементом является порядковый номер кластера, к которому относится i-ая точка
    for i in range(X.shape[0]): # для каждой точки из датасета
        distances = [distance(X[i], center) for center in centers] # массив расстояний от данной i-ой точки до каждого центра
        cluster = np.argmin(distances) # выбираем минимальное из расстояний до центроидов. cluster -- порядковый номер кластера (с нуля), к которому теперь относится i-ая точка
        clusters.append(cluster)
    return clusters

def update_centers(X, clusters, k):
    '''Функция для обновления центров кластеров'''
    centers = np.zeros((k, X.shape[1])) #
    for i in range(k): # сначала для первого, потом для второго, затем для третьего кластера
        cluster_points = [X[j] for j in range(len(X)) if clusters[j] == i] # -- оставляем из датасета X только те точки которые принадлежат i-му кластеру
        center = np.mean(cluster_points, axis=0)
        centers[i] = center
    return centers

def plot_clusters(X, clusters, centers):
    '''Функция для визуализации данных'''
    plt.scatter(X[:, 0], X[:, 12], c=clusters, cmap='viridis') # оси -- первые два столбца
    plt.scatter(centers[:, 0], centers[:, 12], c='black', s=200, alpha=0.5)
    plt.show()

def kmeans(X, k, num_iterations):
    '''Функция для запуска алгоритма K-means'''
    centers = initialize_centers(X, k)
    clusters = []
    for i in range(num_iterations):
        clusters = assign_clusters(X, centers)
        centers = update_centers(X, clusters, k)
        plot_clusters(X, clusters, centers)
    return (centers, clusters)


from datasets import wine as cur_dataset
# from datasets import car as cur_dataset
# from datasets import bank_marketing as cur_dataset

X = normalized(cur_dataset.features()) # датасет
y = normalized(cur_dataset.targets())  # эталонное распределение по кластерам

(centers, clusters) = kmeans(X,3, 25)

# Оценка точности кластеризации
matchings = int(sum([el1 == el2 for (el1, el2) in zip(clusters, y)])) # количество совпадающих с эталоным решением кластеров
accuracy = matchings / len(y)
print('Accuracy:', accuracy)