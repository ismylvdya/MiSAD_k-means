import numpy as np
import os # для экпорта изображений
from datetime import datetime # для экпорта изображений
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap # для своего триколора


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
        centers[i] = np.mean(cluster_points, axis=0) # и для данных точек из i-го кластера считаем среднее
    return centers

# Создание пользовательской цветовой карты триколора
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['orangered', 'royalblue', 'gold'])

def plot_export_and_show(X, clusters, centers, cur_iter, first_axis=0, second_axis=1, save_path=None):
    '''Функция для визуализации данных'''
    X_normalized_features = X.get_normalized_features()

    plt.scatter(X_normalized_features[:, first_axis], X_normalized_features[:, second_axis], c=clusters, cmap=custom_cmap) # X[:, первая ось], X[:, вторая ось]
    plt.scatter(centers[:, first_axis], centers[:, second_axis], c='black', s=400, alpha=0.9)

    # Подпись осей
    plt.xlabel(str(first_axis) + ': ' + str(X.get_axis_name(first_axis)) + ' (normalized)')
    plt.ylabel(str(second_axis) + ': ' + str(X.get_axis_name(second_axis)) + ' (normalized)')

    # Сохранение изображения, если указан путь
    if save_path:
        # Проверка, что папка существует, если нет — создать её
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Полный путь для сохранения
        full_path = os.path.join(save_path, str(cur_iter)+'iter.jpg')

        # Сохранение изображения в формате JPG
        plt.savefig(full_path, format='jpg')

    plt.show()

def create_folder_in(base_path='.'):
    '''создает папку с именем "plots_YYYY-MM-DD_HH-MM-SS" в передаваемой директории'''
    # Получение текущей даты и времени в формате 'YYYY-MM-DD_HH-MM-SS'
    current_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    # Полный путь для новой папки
    full_path = os.path.join(base_path, 'plots_'+current_time)
    # Создание папки
    os.makedirs(full_path, exist_ok=True)

    return full_path


def kmeans(X):
    '''Функция для запуска алгоритма K-means
    возврщает получившиеся на полседней итерации центроиды, распределение точек по кластерам и полученное количество итераций'''

    X_normalized_features = X.get_normalized_features()
    cur_centers = initialize_centers(X_normalized_features, X.k)
    prev_centers = []
    clusters = []
    export_dir = create_folder_in('./images') # куда экспортировать графики с каждой итерации

    i = 0
    while not np.array_equal(prev_centers, cur_centers):
        prev_centers = np.copy(cur_centers)  # сохраняем предыдущие центры для последующего сравнения
        clusters = assign_clusters(X_normalized_features, cur_centers)
        cur_centers = update_centers(X_normalized_features, clusters, X.k)
        plot_export_and_show(X, clusters, cur_centers, i, 0, 12, export_dir)
        i += 1

    return (cur_centers, clusters, i)


def matches_counts_in(clusters, targets, k):
    '''возвращает (количество совпадающих точек, индексы несовпадений) между нашей кластеризацией (на k кластеров) и эталонной С УЧЕТОМ РАЗНОЙ НУМЕРАЦИИ КЛАСТЕРОВ'''

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

    # print(posible_pairs)
    # print(top_k_pairs)
    # print(matches_count)

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
# from datasets import car as cur_dataset
# from datasets import bank_marketing as cur_dataset

Xobject = CurDataset() # класс -- чтобы передавать объект класса в визуализирующую функцию и иметь поэтому доступ к именам осей

for x in range(13):
    for y in range(13):
        if y > x:
            plt.scatter(Xobject.get_normalized_features()[:, x], Xobject.get_normalized_features()[:, y])
            # Полный путь для сохранения
            full_path = os.path.join('./images', f'{x}-{y}.jpg')
            # Сохранение изображения в формате JPG
            plt.savefig(full_path, format='jpg')
            plt.show()


#прогонка kmeans
# (centers, clusters, iter_count) = kmeans(Xobject)
# print(f'\nkmeans() отработал за {iter_count} итераций')

# Оценка точности кластеризации
# targets = Xobject.get_targets()
# (matches_count, diff_indexes) = matches_counts_in(clusters, targets, Xobject.k)
#
# print_with_diff(clusters, targets, diff_indexes)
#
# accuracy = matches_count / len(targets)
#
# print(f'Accuracy:  {matches_count} / {len(targets)}  =  {accuracy:5.3f}')