from ucimlrepo import fetch_ucirepo

# fetch dataset
wine_dataset = fetch_ucirepo(id=109)

def features():
    '''Return features of wine dataset as numpy array (178 объектов (строк) и 13 признаков (столбцов))'''
    return wine_dataset.data.features.to_numpy()  # data импортируется как pandas dataframes поэтому конвертируем в numpy массивы

def targets():
    '''Return targets of wine dataset as numpy array'''
    return wine_dataset.data.targets.to_numpy()  # data импортируется как pandas dataframes поэтому конвертируем в numpy массивы

def print_metadata():
    print(wine_dataset.metadata)

def print_variable_inf():
    print(wine_dataset.variables)