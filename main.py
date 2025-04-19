from torch.utils.data import DataLoader
import torch
from poll_dataset import PollutionDataset
from train_nn import PollutionLSTM, train_model, test_model

batch_size = 32
input_window = 10
hidden_size = 32
n_predictions = 1
n_layers = 1
sgd_lr = 0.001
adam_lr = 0.001
rmsprop_lr = 0.01
adagrad_lr = 0.01
n_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = PollutionDataset(root='./Dataset_Inquinamento', param='NO2', test=False, input_window=input_window, output_window=n_predictions)
test_dataset = PollutionDataset(root='./Dataset_Inquinamento', param='NO2', test=True, input_window=input_window, output_window=n_predictions)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# SGD
sgd_net = PollutionLSTM(input_window, hidden_size, n_predictions, n_layers).to(device)
sgd_optimizer = torch.optim.SGD(params=sgd_net.parameters(), lr=sgd_lr)

# Adam
adam_net = PollutionLSTM(input_window, hidden_size, n_predictions, n_layers).to(device)
adam_optimizer = torch.optim.Adam(params=adam_net.parameters(), lr=adam_lr)

# RMSprop
rmsprop_net = PollutionLSTM(input_window, hidden_size, n_predictions, n_layers).to(device)
rmsprop_optimizer = torch.optim.RMSprop(params=rmsprop_net.parameters(), lr=rmsprop_lr)

# Adagrad
adagrad_net = PollutionLSTM(input_window, hidden_size, n_predictions, n_layers).to(device)
adagrad_optimizer = torch.optim.Adagrad(params=adagrad_net.parameters(), lr=adagrad_lr)

loss_fun = torch.nn.MSELoss()

# training
for epoch in range(n_epochs):
    sgd_loss, sgd_mae = train_model(sgd_net, train_loader, loss_fun, sgd_optimizer, device)
    adam_loss, adam_mae = train_model(adam_net, train_loader, loss_fun, adam_optimizer, device)
    rmsprop_loss, rmsprop_mae = train_model(rmsprop_net, train_loader, loss_fun, rmsprop_optimizer, device)
    adagrad_loss, adagrad_mae = train_model(adagrad_net, train_loader, loss_fun, adagrad_optimizer, device)

    print(f'Epoch: {epoch + 1}/{n_epochs}:')
    print(f'SGD loss:\t{sgd_loss:.6f}\t\tSGD MAE:\t{sgd_mae:.6f}')
    print(f'Adam loss:\t{adam_loss:.6f}\t\tAdam MAE:\t{adam_mae:.6f}')
    print(f'RMSprop loss:\t{rmsprop_loss:.6f}\t\tRMSprop MAE:\t{rmsprop_mae:.6f}')
    print(f'Adagrad loss:\t{adagrad_loss:.6f}\t\tAdagrad MAE:\t{adagrad_mae:.6f}')
