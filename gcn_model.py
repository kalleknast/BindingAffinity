import torch
import sys
import numpy as np
from torch_geometric.loader import DataLoader
from data import BindingAffinityDataset
import matplotlib.pyplot as plt
from models import GCN


dataset_name = 'DAVIS'
root = 'data'
dataset = BindingAffinityDataset(root, dataset_name=dataset_name,
                                 network_type='gcn')
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = GCN(dataset, n_hidden=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


def train(data_loader, verbose=False):

    model.train()
    data_loader.dataset.partition = 'train'

    if verbose:
        print('Batch loss:', end=' ')

    for data in data_loader:    # Iterate in batches over the training dataset.
        out = model(data)       # Perform a single forward pass.
        loss = criterion(out, data[2])  # Compute the loss.
        if verbose:
            s = f'{loss.detach():.3f}'
            print(s, end='\b'*len(s))
            sys.stdout.flush()
        loss.backward()            # Derive gradients.
        optimizer.step()           # Update parameters based on gradients.
        optimizer.zero_grad()      # Clear gradients.

    if verbose:
        print()


# def evaluate(data_loader, partition='valid'):
#
#     data_loader.dataset.partition = partition
#     model.eval()
#     mse = []
#
#     for data in data_loader:  # Iterate in batches over the validation dataset.
#         out = model(data)
#         mse.append(float(((out - data[2])**2).mean().detach()))
#
#     return mse

def evaluate(data_loader, partition='valid'):

    data_loader.dataset.partition = partition
    model.eval()
    mse = []
    pred = []
    y = []

    for data in data_loader:  # Iterate in batches over the validation dataset.
        out = model(data)
        mse.append(float(((out - data[2])**2).mean()))
        pred.extend(list(np.array(out.detach()).flatten()))
        y.extend(list(np.array(data[2].detach())))

    return pred, y, mse


for epoch in range(1, 6):

    train(data_loader)
    pred, y, mse = evaluate(data_loader, partition='train')
    train_mse = np.mean(mse)
    pred, y, mse = evaluate(data_loader, partition='valid')
    valid_mse = np.mean(mse)

    print(f'Epoch: {epoch:02d}, train MSE: {train_mse:.4f}'
          f', valid MSE: {valid_mse:.4f}')

plt.plot(y, '.k', label='y')
plt.plot(pred, '.r', label='pred')
plt.show()
