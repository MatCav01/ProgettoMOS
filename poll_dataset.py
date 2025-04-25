import torch
import numpy
import pandas
import os, sys

class PollutionDataset(torch.utils.data.Dataset):
    def __init__(self, root, param: {'PM10', 'NO2', 'C6H6'} = 'PM10', test = False, input_window = 10, output_window = 1):
        super().__init__()
        self.root = root
        param_id = {
            'PM10': '005_PM10',
            'NO2': '008_NO2',
            'C6H6': '020_C6H6'
        }
        self.param = param_id[param]
        self.test = test
        self.input_window = input_window
        self.output_window = output_window

        self.sequences, self.targets = self.load_data()
    
    def load_data(self):
        self.root = os.path.join(self.root, 'TestSet' if self.test else 'TrainSet', self.param)

        try:
            # concatenation data from all csv files
            data = torch.tensor([], dtype=torch.float32)

            for filename in os.listdir(self.root):
                filepath = os.path.join(self.root, filename)
                dataframe = pandas.read_csv(filepath, sep=',', usecols=['VALORE'], dtype=numpy.float32)

                df_tensor = torch.from_numpy(dataframe.to_numpy().reshape(-1))
                data = torch.cat((data, df_tensor), dim=0)
        
        except FileNotFoundError as err:
            print(err.with_traceback(None), file=sys.stderr)
            sys.exit(1)
        
        # creation input and output sequences
        sequences = torch.tensor([], dtype=torch.float32)
        targets = torch.tensor([], dtype=torch.float32)

        for i in range(len(data) - self.input_window - self.output_window + 1):
            sequence = data[i : i + self.input_window].unsqueeze(0)
            target = data[i + self.input_window : i + self.input_window + self.output_window].unsqueeze(0)
            
            sequences = torch.cat((sequences, sequence), dim=0)
            targets = torch.cat((targets, target), dim=0)

        sequences = sequences.unsqueeze(-1)
        # targets = targets.unsqueeze(-1)

        return sequences, targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]
