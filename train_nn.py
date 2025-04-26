import torch

class PollutionLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_predictions = 1, n_layers = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.lstm.hidden_size, n_predictions)
    
    def forward(self, x):
        outLSTM = self.lstm(x)
        out = self.fc(outLSTM[0][:, -1, :])

        return out
    
def train_model(model: PollutionLSTM,
                train_loader: torch.utils.data.DataLoader,
                loss_fun: torch.nn.MSELoss,
                optimizer: torch.optim.Optimizer,
                device: torch.device):
    model.train()
    train_loss = 0.0
    mae = 0.0

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        predictions = model(sequences)
        loss = loss_fun(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mae += torch.mean(torch.abs(predictions - targets)).item()

    train_loss /= len(train_loader) #* train_loader.batch_size
    mae /= len(train_loader) #* train_loader.batch_size

    return train_loss, mae

def test_model(model: PollutionLSTM,
               test_loader: torch.utils.data.DataLoader,
               loss_fun: torch.nn.MSELoss,
               device: torch.device):
    model.eval()
    test_loss = 0.0
    mae = 0.0

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)
            loss = loss_fun(predictions, targets)

            test_loss += loss.item()
            mae += torch.mean(torch.abs(predictions - targets)).item()

    test_loss /= len(test_loader) #* test_loader.batch_size
    mae /= len(test_loader) #* test_loader.batch_size

    return test_loss, mae
