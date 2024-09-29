from ucimlrepo import fetch_ucirepo
import numpy as np
# для импортирования всевозсожных осей:
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap # для своего триколора

def normalized(data):
    '''Функция для нормализации данных
    принимает на вход двумерный numpy массив
    возвращает нормализованный массив (в нем теперь значения из [0,1])'''

    data_new = np.copy(data)
    for j in range(len(data[0])): # перебираем все строки датасета
        min_val = np.min(data[:, j])
        max_val = np.max(data[:, j])
        for i in range(len(data)):
            data_new[i][j] = (data[i][j] - min_val) / (max_val - min_val)
    return data_new

class WineDataset:
    def __init__(self, dataset_id=109):
        # Загружаем набор данных при инициализации класса
        self.wine_dataset = fetch_ucirepo(id=dataset_id)

        self.k = 3  # Количество кластеров

        # Устанавливаем названия признаков
        self.axis_names = {
            0: 'Alcohol',
            1: 'Malic acid',
            2: 'Ash',
            3: 'Alcalinity of ash',
            4: 'Magnesium',
            5: 'Total phenols',
            6: 'Flavanoids',
            7: 'Nonflavanoid phenols',
            8: 'Proanthocyanins',
            9: 'Color intensity',
            10: 'Hue',
            11: 'OD280/OD315 of diluted wines',
            12: 'Proline'
        }

    def get_features(self):
        '''Возвращает весь датасет как numpy массив'''
        return self.wine_dataset.data.features.to_numpy()

    def get_normalized_features(self):
        '''Возвращает весь датасет НОРМАЛИЗОВАННЫЙ ДО [0,1] как numpy массив'''
        return normalized(self.wine_dataset.data.features.to_numpy())

    def get_targets(self):
        '''Возвращает эталонную кластеризацию как нормальный numpy массив'''
        targets = []
        for el in self.wine_dataset.data.targets.to_numpy():
            targets.append(*el)
        return targets

    def get_axis_name(self, n):
        '''Возвращает название признака по его номеру'''
        return self.axis_names[n] if n in [0,12] else None


    def plot_all_target_axis(self):
        '''экспортирует в ./images/targets_axis_wine графики во всевозможных осях, раскрашенные по targer'''
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['orangered', 'royalblue', 'gold'])

        # Полный путь для новой папки
        dir_path = './images/targets_axis_wine'
        # Создание папки
        os.makedirs(dir_path, exist_ok=True)

        for x in range(13):
            for y in range(13):
                if y > x:
                    plt.scatter(self.get_normalized_features()[:, x], self.get_normalized_features()[:, y], c=self.get_targets(),
                                cmap=custom_cmap)  # X[:, первая ось], X[:, вторая ось]
                    # Сохранение изображения в формате JPG
                    plt.savefig(dir_path + f'/{x}-{y}.jpg', format='jpg')
                    plt.show()  # здесь -- просто для очистки полотна


    def print_metadata(self):
        '''Печатает метаданные набора данных'''
        print(self.wine_dataset.metadata)

    def print_variable_info(self):
        '''Печатает информацию о переменных набора данных'''
        print(self.wine_dataset.variables)