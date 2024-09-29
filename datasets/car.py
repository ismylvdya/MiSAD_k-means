from ucimlrepo import fetch_ucirepo
import numpy as np
# для импортирования всевозсожных осей:
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap # для своего триколора

buying_dict = {'vhigh' : 0,
                'high' : 1,
                 'med' : 2,
                 'low' : 3}
maint_dict = {'vhigh' : 0,
               'high' : 1,
                'med' : 2,
                'low' : 3}
doors_dict = {'2' : 0,
              '3' : 1,
              '4' : 2,
          '5more' : 3}
persons_dict = {'2' : 0,
                '4' : 1,
             'more' : 2}
lug_boot_dict = {'small' : 0,
                   'med' : 1,
                   'big' : 2}
safety_dict = {'low' : 0,
               'med' : 1,
              'high' : 2}
targets_dict = {'unacc' : 0,
                'acc' : 1,
                'good' : 2,
                'vgood' : 3}

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

class CarDataset:
    def __init__(self, dataset_id=19):
        self.dataset = fetch_ucirepo(id=dataset_id)

        self.features = self.dataset.data.features.to_numpy()
        for i in range(len(self.features)):
                self.features[i][0] = buying_dict[str(self.features[i][0])]
                self.features[i][1] = maint_dict[str(self.features[i][1])]
                self.features[i][2] = doors_dict[str(self.features[i][2])]
                self.features[i][3] = persons_dict[str(self.features[i][3])]
                self.features[i][4] = lug_boot_dict[str(self.features[i][4])]
                self.features[i][5] = safety_dict[str(self.features[i][5])]

        self.normalized_features = normalized(self.dataset.data.features.to_numpy())

        self.targets = []
        for el in self.dataset.data.targets.to_numpy():
            self.targets.append(*el)
        for i in range(len(self.targets)):
            self.targets[i] = targets_dict[str(self.targets[i])]

        self.k = 4  # Количество кластеров (задаем из условия поставленной задачи)

        # Устанавливаем названия признаков
        self.axis_names = {
            0: 'buying',
            1: 'maint',
            2: 'doors',
            3: 'persons',
            4: 'lug_boot',
            5: 'safety'
        }

        self.best_axis1 = 0
        self.best_axis2 = 1


    def plot_all_target_axis(self):
        '''экспортирует в ./images/targets_axis_car графики во всевозможных осях, раскрашенные по targer'''
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['orangered', 'royalblue', 'gold'])

        # Полный путь для новой папки
        dir_path = './images/targets_axis_car'
        # Создание папки
        os.makedirs(dir_path, exist_ok=True)

        for x in range(self.features.shape[1]): # obj.features.shape[1] для car равно 6
            for y in range(self.features.shape[1]):
                if y > x:
                    plt.scatter(self.normalized_features[:, x], self.normalized_features[:, y], c=self.targets,
                                cmap=custom_cmap)  # X[:, первая ось], X[:, вторая ось]
                    # Сохранение изображения в формате JPG
                    plt.savefig(dir_path + f'/{x}-{y}.jpg', format='jpg')
                    plt.show()  # здесь -- просто для очистки полотна



    def print_metadata(self):
        '''Печатает метаданные набора данных'''
        print(self.dataset.metadata)

    def print_variable_info(self):
        '''Печатает информацию о переменных набора данных'''
        print(self.dataset.variables)
