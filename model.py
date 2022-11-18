import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, Linear
from torch_geometric.nn import global_mean_pool
from data import BindingAffinityDataset


class GCN(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(GCN, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)
        self.dense1 = Linear(dataset.num_node_features, n_hidden)
        self.dense2 = Linear(n_hidden, n_hidden)
        self.conv1 = GCNConv(n_hidden, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.dense3 = Linear(2*n_hidden, 256)
        self.dense4 = Linear(256, 128)
        self.dense5 = Linear(128, 1)

    def forward(self, data):
        # Drug
        # average pooling
        x0, edge_index = data[0].x, data[0].edge_index
        x0 = self.batchnorm(x0)
        x0 = self.dense1(x0).relu()
        x0 = self.dense2(x0).relu()
        x0 = self.conv1(x0, edge_index).relu()
        x0 = self.conv2(x0, edge_index).relu()
        x0 = self.conv3(x0, edge_index).relu()
        x0 = global_mean_pool(x0, data[0].batch)
        # x0 = torch.cat(x0, dim=0)

        # Protein
        # average pooling
        x1, edge_index = data[1].x, data[1].edge_index
        x1 = self.batchnorm(x1)
        x1 = self.dense1(x1).relu()
        x1 = self.dense2(x1).relu()
        x1 = self.conv1(x1, edge_index).relu()
        x1 = self.conv2(x1, edge_index).relu()
        x1 = self.conv3(x1, edge_index).relu()
        x1 = global_mean_pool(x1, data[1].batch)
        # x1 = torch.cat(x1, dim=0)

        x = torch.cat([x0, x1], dim=1)
        x = self.dense3(x).relu()
        x = self.dense4(x).relu()
        x = self.dense5(x)

        return x


dataset_name = 'DAVIS'
root = 'data'
dataset = BindingAffinityDataset(root, dataset_name=dataset_name)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = GCN(dataset, n_hidden=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


def train(data_loader):

    model.train()
    data_loader.dataset.partition = 'train'

    for data in data_loader:  # Iterate in batches over the training dataset.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data[-1])  # Compute the loss.
        print(f'{loss.detach():.3f}', end=' ')
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    print()


def evaluate(data_loader, partition='valid'):

    data_loader.dataset.partition = partition
    model.eval()
    mse = []

    for data in data_loader:  # Iterate in batches over the validation dataset.
        out = model(data)
        mse.append(float(((out - data[-1])**2).mean().detach()))

    return mse


for epoch in range(1, 11):

    train(data_loader)
    valid_mse = np.mean(evaluate(data_loader, partition='valid'))
    train_mse = np.mean(evaluate(data_loader, partition='train'))

    print(f'Epoch: {epoch:03d}, train MSE: {valid_mse:.4f}'
          f', valid MSE: {valid_mse:.4f}')
