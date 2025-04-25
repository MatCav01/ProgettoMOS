import matplotlib.pyplot as plt

def plot_losses(sgd_loss, adam_loss, rmsprop_loss, adagrad_loss):
    plt.figure(figsize=(12, 8))

    plt.plot(sgd_loss, label='SGD', color='r', marker='o')
    plt.plot(adam_loss, label='Adam', color='b', marker='o')
    plt.plot(rmsprop_loss, label='RMSprop', color='g', marker='o')
    plt.plot(adagrad_loss, label='Adagrad', color='m', marker='o')

    plt.title('MSE convergence with SGD, Adam, RMSprop and Adagrad optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')

    plt.legend()
    plt.grid(True)
    plt.show()
