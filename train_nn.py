import torch

class PollutionLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1)
        self.fc = torch.nn.Linear(self.lstm.hidden_size, output_size)
    
    def forward(self, x):
        outLSTM = self.lstm(x)
        print(outLSTM)
        out = self.fc(outLSTM)

        return out

# for debugging
if __name__ == '__main__':
    from poll_dataset import PollutionDataset
    dataset = PollutionDataset(root='./Dataset', param='PM10')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = PollutionLSTM(input_size=1, hidden_size=1, output_size=1)
    for x, y in dataloader:
        output = model(x)
        print(output)
