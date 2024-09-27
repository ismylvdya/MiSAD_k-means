from ucimlrepo import fetch_ucirepo

# fetch dataset 
bank_marketing_dataset = fetch_ucirepo(id=222)

def features():
    '''Return features of wine dataset as numpy array (45211 объектов (строк) и 16 признаков (столбцов))'''
    return bank_marketing_dataset.data.features.to_numpy()  # data импортируется как pandas dataframes поэтому конвертируем в numpy массивы

def targets():
    '''Return targets of wine dataset as numpy array'''
    return bank_marketing_dataset.data.targets.to_numpy()  # data импортируется как pandas dataframes поэтому конвертируем в numpy массивы

def print_metadata():
    print(bank_marketing_dataset.metadata)

def print_variable_inf():
    print(bank_marketing_dataset.variables)