import torch
import sys
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, Linear
from torch_geometric.nn import global_mean_pool
from data import BindingAffinityDataset


class MLP(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(MLP, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)
        self.dense1 = Linear(dataset.num_node_features, 2*n_hidden)
        self.dense2 = Linear(2*n_hidden, n_hidden)
        self.dense3 = Linear(2*n_hidden, 256)
        self.dense4 = Linear(256, 128)
        self.dense5 = Linear(128, 1)

    def forward(self, data):
        # Drug
        # average pooling
        x0 = data[0].x
        x0 = self.batchnorm(x0)
        x0 = self.dense1(x0).relu()
        x0 = self.dense2(x0).relu()
        x0 = global_mean_pool(x0, data[0].batch)
        # x0 = torch.cat(x0, dim=0)

        # Protein
        # average pooling
        x1 = data[1].x
        x1 = self.batchnorm(x1)
        x1 = self.dense1(x1).relu()
        x1 = self.dense2(x1).relu()
        x1 = global_mean_pool(x1, data[1].batch)
        # x1 = torch.cat(x1, dim=0)

        x = torch.cat([x0, x1], dim=1)
        x = self.dense3(x).relu()
        x = self.dense4(x).relu()
        x = self.dense5(x)

        return x


dataset_name = 'DAVIS'
root = 'data'
network_type = 'mlp'
dataset = BindingAffinityDataset(root, dataset_name=dataset_name,
                                 network_type=network_type)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MLP(dataset, n_hidden=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


def train(data_loader, verbose=False):

    model.train()
    data_loader.dataset.partition = 'train'

    for data in data_loader:  # Iterate in batches over the training dataset.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data[-1])  # Compute the loss.
        if verbose:
            print(f'{loss.detach():.3f}', end=' ')
            sys.stdout.flush()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def evaluate(data_loader, partition='valid'):

    data_loader.dataset.partition = partition
    model.eval()
    mse = []

    for data in data_loader:  # Iterate in batches over the validation dataset.
        out = model(data)
        mse.append(float(((out - data[-1])**2).mean().detach()))

    return mse


for epoch in range(1, 101):

    train(data_loader)
    train_mse = np.mean(evaluate(data_loader, partition='train'))
    valid_mse = np.mean(evaluate(data_loader, partition='valid'))

    print(f'Epoch: {epoch:03d}, train MSE: {train_mse:.4f}'
          f', valid MSE: {valid_mse:.4f}')
