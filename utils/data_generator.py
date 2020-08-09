import pandas as pd
import os


class DataGenerator:

    def __init__(self):
        data_dir = './data'
        self.data_folders_list = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        self.data_folders_list.sort()

        models_dir = './models'
        self.models_list = [f.path for f in os.scandir(models_dir) if f.is_file()]
        self.models_list.sort()

    def training_generator(self):
        for folder_name in self.data_folders_list:
            yield pd.read_csv('./data/' + folder_name + '/train.csv'), folder_name

    def evaluation_generator(self):
        for folder_name in self.data_folders_list:
            yield pd.read_csv('./data/' + folder_name + '/test.csv'), folder_name

    def models_generator(self):
        for model in self.models_list:
            yield model
