import torch
from torch_geometric.nn import BatchNorm, Linear, GCNConv
from torch_geometric.nn import global_mean_pool


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
        x0 = global_mean_pool(x0, data[0].batch)
        x0 = self.batchnorm(x0)
        x0 = self.dense1(x0).relu()
        x0 = self.dense2(x0).relu()
        # x0 = torch.cat(x0, dim=0)

        # Protein
        # average pooling
        x1 = data[1].x
        x1 = global_mean_pool(x1, data[1].batch)
        x1 = self.batchnorm(x1)
        x1 = self.dense1(x1).relu()
        x1 = self.dense2(x1).relu()
        # x1 = torch.cat(x1, dim=0)

        x = torch.cat([x0, x1], dim=1)
        x = self.dense3(x).relu()
        x = self.dense4(x).relu()
        x = self.dense5(x)

        return x


class GCN(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(GCN, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)
        self.dense1 = Linear(dataset.num_node_features, 2*n_hidden)
        self.dense2 = Linear(2*n_hidden, n_hidden)
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
