import matplotlib.pyplot as plt

def plot_losses(n_epochs, sgd_loss, adam_loss, rmsprop_loss, adagrad_loss):
    plt.figure(figsize=(12, 8))

    x = list(range(1, n_epochs + 1))
    plt.plot(x, sgd_loss, label='SGD', color='r', marker='o')
    plt.plot(x, adam_loss, label='Adam', color='b', marker='o')
    plt.plot(x, rmsprop_loss, label='RMSprop', color='g', marker='o')
    plt.plot(x, adagrad_loss, label='Adagrad', color='m', marker='o')

    plt.title('MSE convergence with SGD, Adam, RMSprop and Adagrad optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')

    plt.legend()
    plt.grid(True)
    plt.show()
