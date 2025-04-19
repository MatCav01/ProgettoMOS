import torch

class PollutionLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_predictions = 1, n_layers = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.lstm.hidden_size, n_predictions)
    
    def forward(self, x):
        outLSTM = self.lstm(x)
        print(outLSTM)
        out = self.fc(outLSTM[:, -1, :])

        return out
    
def train_model(model, train_loader, loss_fun, optimizer, device):
    model.train()
    train_loss = 0.0
    mae = 0.0

    for sequences, targets in train_loader:
        sequences.to(device)
        targets.to(device)

        predictions = model(sequences)
        loss = loss_fun(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mae += torch.mean(torch.abs(predictions - targets))

    train_loss /= len(train_loader)
    mae /= len(train_loader)

    return train_loss, mae

def test_model(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0.0
    mae = 0.0

    for sequences, targets in test_loader:
        sequences.to(device)
        targets.to(device)

        predictions = model(sequences)
        loss = loss_fun(predictions, targets)

        test_loss += loss.item()
        mae += torch.mean(torch.abs(predictions - targets))

    test_loss /= len(test_loader)
    mae /= len(test_loader)

    return test_loss, mae


# for debugging
if __name__ == '__main__':
    from poll_dataset import PollutionDataset
    dataset = PollutionDataset(root='./Dataset', param='PM10')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = PollutionLSTM(input_size=1, hidden_size=32, n_predictions=1, num_layers=1)
    for x, y in dataloader:
        output = model(x)
        print(output)
