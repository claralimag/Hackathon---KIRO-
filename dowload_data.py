import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt

# télécharger les insatances depuis les fichiers csv
def download_data(file_path):
    data = pd.read_csv(file_path)
    return data

dataset_1 = download_data('/workspaces/Hackathon---KIRO-/instances/instance_01.csv')
vehicles = download_data('/workspaces/Hackathon---KIRO-/instances/vehicles.csv')

# Look for missing data :
print(dataset_1.info())

#Missing data handling : order_weight, window_start, window_end : 
#Si id = 0 alors c'est le dépot, order_weight, window_start_window_end = 0

dataset_1['order_weight'] = np.where(dataset_1['id'] == 0, 0, dataset_1['order_weight'].fillna(0))
dataset_1['window_start'] = np.where(dataset_1['id'] == 0, 0, dataset_1['window_start'].fillna(0))
dataset_1['window_end'] = np.where(dataset_1['id'] == 0, 0, dataset_1['window_end'].fillna(0))

print(dataset_1.info())
print(dataset_1.head())



