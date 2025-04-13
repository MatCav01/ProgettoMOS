import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size n_labels):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=)