from ucimlrepo import fetch_ucirepo

# fetch dataset 
car_evaluation_dataset = fetch_ucirepo(id=19) 

def features():
    '''Return features of wine dataset as numpy array (1728 объектов (строк) и 6 признаков (столбцов))'''
    return car_evaluation_dataset.data.features.to_numpy()  # data импортируется как pandas dataframes поэтому конвертируем в numpy массивы

def targets():
    '''Return targets of wine dataset as numpy array'''
    return car_evaluation_dataset.data.targets.to_numpy()  # data импортируется как pandas dataframes поэтому конвертируем в numpy массивы

def print_metadata():
    print(car_evaluation_dataset.metadata)

def print_variable_inf():
    print(car_evaluation_dataset.variables)