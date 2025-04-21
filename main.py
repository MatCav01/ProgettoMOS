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

# weights = input('Do you want to train new models or load existing parameters? (train or load) ')
# if weights == 'train':
# training
for epoch in range(n_epochs):
    sgd_train_loss, sgd_train_mae = train_model(sgd_net, train_loader, loss_fun, sgd_optimizer, device)
    adam_train_loss, adam_train_mae = train_model(adam_net, train_loader, loss_fun, adam_optimizer, device)
    rmsprop_train_loss, rmsprop_train_mae = train_model(rmsprop_net, train_loader, loss_fun, rmsprop_optimizer, device)
    adagrad_train_loss, adagrad_train_mae = train_model(adagrad_net, train_loader, loss_fun, adagrad_optimizer, device)

    print(f'Epoch: {epoch + 1}/{n_epochs}:')
    print(f'SGD loss:\t{sgd_train_loss:.6f}\t\tSGD MAE:\t{sgd_train_mae:.6f}')
    print(f'Adam loss:\t{adam_train_loss:.6f}\t\tAdam MAE:\t{adam_train_mae:.6f}')
    print(f'RMSprop loss:\t{rmsprop_train_loss:.6f}\t\tRMSprop MAE:\t{rmsprop_train_mae:.6f}')
    print(f'Adagrad loss:\t{adagrad_train_loss:.6f}\t\tAdagrad MAE:\t{adagrad_train_mae:.6f}')
    
#     save_params = input('Do you want to save the models parameters? (yes or anything else) ')
#     if save_params == 'yes':
#         torch.save(sgd_net.state_dict(), './SavedWeights/sgd_net_weights.pt')
#         torch.save(adam_net.state_dict(), './SavedWeights/adam_net_weights.pt')
#         torch.save(rmsprop_net.state_dict(), './SavedWeights/rmsprop_net_weights.pt')
#         torch.save(adagrad_net.state_dict(), './SavedWeights/adagrad_net_weights.pt')

# elif weights == 'load':
#     sgd_net.load_state_dict(torch.load('./SavedWeights/sgd_net_weights.pt'))
#     adam_net.load_state_dict(torch.load('./SavedWeights/adam_net_weights.pt'))
#     rmsprop_net.load_state_dict(torch.load('./SavedWeights/rmsprop_net_weights.pt'))
#     adagrad_net.load_state_dict(torch.load('./SavedWeights/adagrad_net_weights.pt'))

sgd_test_loss, sgd_test_mae = test_model(sgd_net, test_loader, loss_fun, sgd_optimizer, device)
adam_test_loss, adam_test_mae = test_model(adam_net, test_loader, loss_fun, adam_optimizer, device)
rmsprop_test_loss, rmsprop_test_mae = test_model(rmsprop_net, test_loader, loss_fun, rmsprop_optimizer, device)
adagrad_test_loss, adagrad_test_mae = test_model(adagrad_net, test_loader, loss_fun, adagrad_optimizer, device)
