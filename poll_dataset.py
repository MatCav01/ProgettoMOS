from torch.utils.data import Dataset
import torch
import numpy
import pandas
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

        self.data, self.targets = self.load_data()
    
    def load_data(self):
        self.root = os.path.join(self.root, self.param)

        try:
            data = numpy.array([], dtype=numpy.float32)

            for filename in os.listdir(self.root):
                filepath = os.path.join(self.root, filename)
                dataframe = pandas.read_csv(filepath, sep=',', usecols=['VALORE'], dtype=numpy.float32)
                data = numpy.concatenate([data, dataframe.to_numpy().reshape(-1)], axis=0)
            
            targets = data[1 :]
            data = data[0 : len(data) - 2]

            data = torch.tensor(data, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            return data, targets
        
        except FileNotFoundError as err:
            print(err.with_traceback(None), file=sys.stderr)
            sys.exit(1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# for debugging
if __name__ == '__main__':
    poll = PollutionDataset(root='./Dataset', param='PM10')