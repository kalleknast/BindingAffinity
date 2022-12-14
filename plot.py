import matplotlib.pyplot as plt
import numpy as np
import json


def plot_predictions(y_train, pred_train, y_valid,
                     pred_valid, model_name, epochs):
    '''
    '''
    plt.ion()

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(f'{model_name} - predictions')

    ax = fig.add_subplot(121)
    ax.plot(y_valid, '.k', label='y')
    ax.plot(pred_valid, '.r', label='pred')
    ax.set_ylim([9, 16])
    ax.set_title('Validation set')

    ax = fig.add_subplot(122)
    ax.plot(y_train[:500], '.k', label='y')
    ax.plot(pred_train[:500], '.r', label='pred')
    ax.set_ylim([9, 16])
    ax.set_title('Training set')

    fig.savefig(f'fig/predictions_{model_name}_epochs{epochs}.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_train, pred_train, marker='.', label='training set')
    ax.scatter(y_valid, pred_valid, marker='.', label='validation set')
    ax.legend()
    ax.set_ylim([9, 16])
    ax.set_xlim([9, 16])
    ax.set_ylabel('Predicted affinity')
    ax.set_xlabel('Measured affinity (ground truth)')
    ax.set_title(f'{model_name} - predictions vs ground truth')

    fig.savefig(f'fig/predictions_scatter_{model_name}_epochs{epochs}.png')


def plot_losses(history, model_name, ax=None):
    '''
    '''
    if not ax:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        save = True
    else:
        save = False

    epochs = range(1, len(history['train_loss'])+1)
    ax.plot(epochs, history['train_loss'], label='train')
    ax.plot(epochs, history['valid_loss'], label='valid')
    ax.text(len(history['train_loss'])*.55, .7,
            f"min validation loss: {np.min(history['valid_loss']):.4}")
    ax.set_ylim([0, 1])
    ax.set_ylabel('loss (MSE)')
    ax.set_xlabel('epoch')
    ax.set_title(f'{model_name} - training and validation loss')

    if save:
        fig.savefig(f'fig/training_history_{model_name}_'
                    f'epochs{max(epochs)}.png')


def plot_loss_comparison(fnames, names, title):

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for i, fname in enumerate(fnames):
        history = json.load(open(fname))
        epochs = range(1, len(history['train_loss'])+1)
        ax.plot(epochs, history['train_loss'], ls=':',
                label=f'{names[i]} (train)')
        color = ax.get_lines()[-1].get_color()
        ax.plot(epochs, history['valid_loss'], color=color, ls='-',
                label=f'{names[i]} (valid)')
    ax.plot([epochs[0], epochs[-1]], [0.71, 0.71], 'k',
            label='Predicting the dataset average')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('loss (MSE)')
    ax.set_xlabel('Epoch')

    fig.savefig(f'fig/history_comparision_{title}.png')
