import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap # для своего триколора
# для экпорта изображений
import os
from datetime import datetime


def distance(x1, x2):
    '''возвращает евклидово расстояние'''
    return np.sqrt(np.sum((x1 - x2)**2))

def initialize_centers(X, k):
    '''Функция для инициализации центров кластеров
    возвращает двумерный массив: центроид x его координаты'''
    centers = np.zeros((k, X.shape[1])) # двумерный массив (количество центроидов x координаты каждого). X.shape[1] -- количество столбцов в датасете (количество осей на графике)
    for i in range(k):
        centers[i] = X[np.random.choice(range(X.shape[0]))] # каждый из k кластеров инициализируем координатами случайной точки из датасета
    return centers

def assign_clusters(X, centers):
    '''Функция для распределения точек по кластерам
    возвращает массив распределения'''
    clusters = [] # массив распределения точек по кластерам. Он длины количества точек и каждый его элементом является порядковый номер кластера (с 0), к которому относится i-ая точка
    for i in range(X.shape[0]): # для каждой точки из датасета
        distances = [distance(X[i], center) for center in centers] # массив расстояний от данной i-ой точки до каждого центра
        cluster = np.argmin(distances) # выбираем минимальное из расстояний до центроидов. cluster -- порядковый номер кластера (с нуля), к которому теперь относится i-ая точка
        clusters.append(cluster)
    return clusters

def update_centers(X, clusters, k):
    '''Функция для обновления центров кластеров через вычисление центра мас'''
    centers = np.zeros((k, X.shape[1])) #
    for i in range(k): # сначала для первого, потом для второго, затем для третьего кластера
        cluster_points = [X[j] for j in range(len(X)) if clusters[j] == i] # -- оставляем из датасета X только те точки которые принадлежат i-му кластеру
        if len(cluster_points) > 0:  # Проверяем, есть ли точки в кластере. Если есть то пересчитываем его цетроиду как центр масс
            centers[i] = np.mean(cluster_points, axis=0) # и для данных точек из i-го кластера считаем среднее. axis=0 -- усредняем каждый столбец а не каждую строку

    return centers

# Создание пользовательской цветовой карты триколора
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['orangered', 'royalblue', 'gold'])

def plot_export_and_show(X, clusters, centers, cur_iter, save_path=None, is_iter_last=False):
    '''Функция для экспорта графика clusters и centers на данной cur_iter-ации в save_path-директорию и его визуализации внутри пайчарма'''

    # вычисление порядковых номеров точек (нужно только diff_indexes) на которых кластеризация прошла неверно
    (matches_count, diff_indexes) = matches_counts_in(clusters, X.targets, Xobject.k)

    # визуализация ВСЕХ кластеризированных точек
    plt.scatter(X.normalized_features[:, X.best_axis1], X.normalized_features[:, X.best_axis2], c=clusters, cmap=custom_cmap) # X[:, первая ось], X[:, вторая ось]

    # Визуализация черных точек, которые кластеризовались неверно относително targets
    if is_iter_last == True: # только если это последняя итерация k-means
        for i in diff_indexes:
            plt.scatter(X.normalized_features[i, X.best_axis1], X.normalized_features[i, X.best_axis2], color='black') # X[:, первая ось], X[:, вторая ось]

    # визализация центроидов
    plt.scatter(centers[:, X.best_axis1], centers[:, X.best_axis2], c='black', s=400, alpha=0.7)

    # Подпись осей
    plt.xlabel(str(X.best_axis1) + ': ' + str(X.axis_names[X.best_axis1]) + ' (normalized)')
    plt.ylabel(str(X.best_axis2) + ': ' + str(X.axis_names[X.best_axis2]) + ' (normalized)')

    # Сохранение изображения, если указан путь
    if save_path:
        # Проверка, что папка существует, если нет — создать её
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Полный путь для сохранения
        full_path = os.path.join(save_path, str(cur_iter)+'iter.jpg')

        # Сохранение изображения в формате JPG
        plt.savefig(full_path, format='jpg')

    plt.show() # в т.ч. для очистки полотна

def create_folder_in(base_path='.'):
    '''создает папку с именем "plots_DD-MM-YYYY_HH-MM-SS" в передаваемой директории'''
    # Получение текущей даты и времени в формате 'YYYY-MM-DD_HH-MM-SS'
    current_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    # Полный путь для новой папки
    full_path = os.path.join(base_path, 'plots_'+current_time)
    # Создание папки
    os.makedirs(full_path, exist_ok=True)

    return full_path



def kmeans(X):
    '''Функция прогонки алгоритма K-means
    возврщает получившиеся на полседней итерации центроиды, массив распределения точек по кластерам и полученное количество итераций'''

    cur_centers = initialize_centers(X.normalized_features, X.k)
    prev_centers = []
    clusters = []
    export_dir = create_folder_in('./images') # куда экспортировать графики с каждой итерации

    i = 0
    while not np.array_equal(prev_centers, cur_centers):
        prev_centers = np.copy(cur_centers)  # сохраняем предыдущие центры для последующего сравнения
        clusters = assign_clusters(X.normalized_features, cur_centers)
        cur_centers = update_centers(X.normalized_features, clusters, X.k)
        if not np.array_equal(prev_centers, cur_centers):
            plot_export_and_show(X, clusters, cur_centers, i, export_dir, is_iter_last=False)
        else:
            plot_export_and_show(X, clusters, cur_centers, i, export_dir, is_iter_last=True)
        i += 1

    return (cur_centers, clusters, i)


def matches_counts_in(clusters, targets, k):
    '''возвращает tuple (количество совпадающих точек, индексы несовпадений) между нашей кластеризацией (на k кластеров) и эталонной С УЧЕТОМ РАЗНОЙ НУМЕРАЦИИ КЛАСТЕРОВ'''

    matches_count = 0
    diff_indexes = []

    posible_pairs = {} # словарь типа {(элемент_из_clusters, элемент_из_targets) : ск_раз_встретилась_эта_пара}

    for i in range(len(clusters)):
        cur_pair = (clusters[i], targets[i])
        if cur_pair not in posible_pairs:
            posible_pairs[(clusters[i], targets[i])] = 1
        else:
            posible_pairs[(clusters[i], targets[i])] += 1

    top_k_pairs = sorted(posible_pairs, key=posible_pairs.get, reverse=True)[:k] # -- массив из тех K пар кластеров, которые встретились чаще других

    for i in range(len(clusters)):
        cur_pair = (clusters[i], targets[i])
        if cur_pair in top_k_pairs:
            matches_count += 1
        else:
            diff_indexes.append(i)

    return (matches_count, diff_indexes)


def print_with_diff(clusters, targets, diff_indexes):
    '''выводит clusters, под ним targets. При этом элементы которые по мнению matches_counts_in() не совпадают выделяет красным'''
    print('our clusters: [', end='')
    for i, el in enumerate(clusters):
        if i in diff_indexes:
            print(f'\033[31m{el}\033[0m', end=', ')
        else:
            print(el, end=', ')
    print(']')

    print('     targets: [', end='')
    for i, el in enumerate(targets):
        if i in diff_indexes:
            print(f'\033[31m{el}\033[0m', end=', ')
        else:
            print(el, end=', ')
    print(']')



from datasets.wine import WineDataset as CurDataset

Xobject = CurDataset() # класс -- чтобы передавать объект класса в визуализирующую функцию и иметь поэтому доступ к именам осей

#прогонка kmeans
(centers, clusters, iter_count) = kmeans(Xobject)
print(f'\nkmeans() отработал за {iter_count} итераций')

# Оценка точности кластеризации
targets = Xobject.targets
(matches_count, diff_indexes) = matches_counts_in(clusters, targets, Xobject.k)

print_with_diff(clusters, targets, diff_indexes)

accuracy = matches_count / len(targets)

print(f'Accuracy:  {matches_count} / {len(targets)}  =  {accuracy:5.3f}')