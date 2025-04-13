from torch.utils.data import Dataset
import torch
import pandas as pd
import os, sys

class PollutionDataset(Dataset):
    def __init__(self, root, param: {'PM10', 'NO2', 'C6H6'} = 'PM10'):
        super().__init__()
        self.root = root
        param_id = {
            'PM10': '005_PM10',
            'NO2': '008_NO2',
            'C6H6': '020_C6H6'
        }
        self.param = param_id[param]

        data = self.load_data()
    
    def load_data(self):
        self.root = os.path.join(self.root, self.param)

        try:
            data = []
            for filename in os.listdir(self.root):
                filepath = os.path.join(self.root, filename)
                dataframe = pd.read_csv(filepath, sep=',', usecols=['DATA_INIZIO', 'VALORE'])
                data.append(dataframe.to_numpy())
        
        except FileNotFoundError as err:
            print(err.with_traceback(None), file=sys.stderr)
            sys.exit(1)
        
        return data
