import torch
from torch_geometric.nn import Linear as GraphLinear
from torch_geometric.nn import global_mean_pool, BatchNorm, GCNConv
from torch.nn import Embedding, Conv1d, Conv2d, Linear, Dropout
from torch.nn import AdaptiveMaxPool1d, AdaptiveMaxPool2d


class CNN2D(torch.nn.Module):

    def __init__(self, dataset):
        super(CNN2D, self).__init__()

        # Encode SMILES
        self.embedding_d = Embedding(num_embeddings=dataset.len_drug_vocab + 1,
                                     embedding_dim=128)
        self.conv2d_d1 = Conv2d(1, out_channels=32,
                                kernel_size=(4, 4), padding='valid')
        self.conv2d_d2 = Conv2d(32, out_channels=64,
                                kernel_size=(4, 4), padding='valid')
        self.conv2d_d3 = Conv2d(64, out_channels=96,
                                kernel_size=(4, 4), padding='valid')
        self.max_pool_d = AdaptiveMaxPool2d((1, 119))

        # Encode protein
        self.embedding_p = Embedding(num_embeddings=dataset.len_prot_vocab + 1,
                                     embedding_dim=128)
        self.conv2d_p1 = Conv2d(1, out_channels=32, kernel_size=(8, 4),
                                stride=(2, 1), padding='valid')
        self.conv2d_p2 = Conv2d(32, out_channels=64, kernel_size=(12, 4),
                                stride=(3, 1), padding='valid')
        self.conv2d_p3 = Conv2d(64, out_channels=96, kernel_size=(12, 4),
                                stride=(3, 1), padding='valid')
        self.max_pool_p = AdaptiveMaxPool2d((1, 119))

        # Combined regressor
        # self.dense_1 = Linear(2 * 96, 1024)
        self.dense_1 = Linear(22848, 1024)
        self.dropout_1 = Dropout(p=0.1)
        self.dense_2 = Linear(1024, 1024)
        self.dropout_2 = Dropout(p=0.1)
        self.dense_3 = Linear(1024, 512)
        self.dense_4 = Linear(512, 1)
        torch.nn.init.normal_(self.dense_4.weight)

    def forward(self, xd, xp):
        # Encode drugs/SMILES
        xd = self.embedding_d(xd.int()).reshape((-1, 1, 100, 128))
        xd = self.conv2d_d1(xd).relu()
        xd = self.conv2d_d2(xd).relu()
        xd = self.conv2d_d3(xd).relu()
        xd = self.max_pool_d(xd).squeeze()

        # Encode proteins
        xp = self.embedding_p(xp.int()).reshape((-1, 1, 1000, 128))
        xp = self.conv2d_p1(xp).relu()
        xp = self.conv2d_p2(xp).relu()
        xp = self.conv2d_p3(xp).relu()
        xp = self.max_pool_p(xp).squeeze()

        # Common regression head
        x = torch.cat([xd.reshape((-1, 96 * 119)), xp.reshape((-1, 96 * 119))], dim=1)
        # import ipdb; ipdb.set_trace()
        x = self.dense_1(x).relu()
        x = self.dropout_1(x).relu()
        x = self.dense_2(x).relu()
        x = self.dropout_2(x).relu()
        x = self.dense_3(x).relu()
        pred = self.dense_4(x)

        return pred


class DeepDTA(torch.nn.Module):

    def __init__(self, dataset):
        super(DeepDTA, self).__init__()

        # Encode SMILES
        self.embedding_d = Embedding(num_embeddings=dataset.len_drug_vocab + 1,
                                     embedding_dim=128)
        self.conv1d_d1 = Conv1d(128, out_channels=32,
                                kernel_size=4, padding='valid')
        self.conv1d_d2 = Conv1d(32, out_channels=64,
                                kernel_size=4, padding='valid')
        self.conv1d_d3 = Conv1d(64, out_channels=96,
                                kernel_size=4, padding='valid')
        self.global_max_pool_d = AdaptiveMaxPool1d(1)

        # Encode protein
        self.embedding_p = Embedding(num_embeddings=dataset.len_prot_vocab + 1,
                                     embedding_dim=128)
        self.conv1d_p1 = Conv1d(128, out_channels=32,
                                kernel_size=8, padding='valid')
        self.conv1d_p2 = Conv1d(32, out_channels=64,
                                kernel_size=8, padding='valid')
        self.conv1d_p3 = Conv1d(64, out_channels=96,
                                kernel_size=8, padding='valid')
        self.global_max_pool_p = AdaptiveMaxPool1d(1)

        # Combined regressor
        self.dense_1 = Linear(2 * 96, 1024)
        self.dropout_1 = Dropout(p=0.1)
        self.dense_2 = Linear(1024, 1024)
        self.dropout_2 = Dropout(p=0.1)
        self.dense_3 = Linear(1024, 512)
        self.dense_4 = Linear(512, 1)
        torch.nn.init.normal_(self.dense_4.weight)

    def forward(self, xd, xp):

        # Encode drugs/SMILES
        xd = self.embedding_d(xd.int()).transpose(1, 2)
        xd = self.conv1d_d1(xd).relu()
        xd = self.conv1d_d2(xd).relu()
        xd = self.conv1d_d3(xd).relu()
        xd = self.global_max_pool_d(xd).squeeze()

        # Encode proteins
        xp = self.embedding_p(xp.int()).transpose(1, 2)
        xp = self.conv1d_p1(xp).relu()
        xp = self.conv1d_p2(xp).relu()
        xp = self.conv1d_p3(xp).relu()
        xp = self.global_max_pool_p(xp).squeeze()

        # Common regression head
        x = torch.cat([xd, xp], dim=1)
        x = self.dense_1(x).relu()
        x = self.dropout_1(x).relu()
        x = self.dense_2(x).relu()
        x = self.dropout_2(x).relu()
        x = self.dense_3(x).relu()
        pred = self.dense_4(x)

        return pred


class MLP(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(MLP, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)
        self.dense1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense2 = GraphLinear(2*n_hidden, n_hidden)
        self.dense3 = GraphLinear(2*n_hidden, 256)
        self.dense4 = GraphLinear(256, 128)
        self.dense5 = GraphLinear(128, 1)

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


class BertCNN(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(GCN, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)

        # Drug branch
        self.dense_d1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_d2 = GraphLinear(2*n_hidden, n_hidden)
        self.conv_d1 = Conv1d(n_hidden, 32, kernel_size=4)
        self.conv_d2 = Conv1d(32, 64, kernel_size=6)
        self.conv_d3 = Conv1d(64, 96, kernel_size=8)

        # Protein branch
        self.dense_p1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_p2 = GraphLinear(2*n_hidden, n_hidden)
        self.conv_p1 = Conv1d(n_hidden, 32, kernel_size=4)
        self.conv_p2 = Conv1d(32, 64, kernel_size=6)
        self.conv_p3 = Conv1d(64, 96, kernel_size=8)

        # Common regressor
        self.dense_3 = GraphLinear(2*n_hidden, 256)
        self.dense_4 = GraphLinear(256, 128)
        self.dense_5 = GraphLinear(128, 1)

    def forward(self, data):

        # Drug
        x0 = global_mean_pool(data[0].x, data[0].batch)
        x0 = self.batchnorm(x0)
        x0 = self.dense_d1(x0).relu()
        x0 = self.dense_d2(x0).relu()
        residual = x0
        x0 = self.conv_d1(x0).relu()
        x0 += residual
        residual = x0
        x0 = self.conv_d2(x0).relu()
        x0 += residual
        residual = x0
        x0 = self.conv_d3(x0).relu()
        x0 += residual
        x0 = global_mean_pool(x0, data[0].batch)
        # x0 = torch.cat(x0, dim=0)

        # Protein
        x1 = global_mean_pool(data[1].x, data[1].batch)
        x1 = self.batchnorm(x1)
        x1 = self.dense_p1(x1).relu()
        x1 = self.dense_p2(x1).relu()
        residual = x1
        x1 = self.conv_p1(x1).relu()
        x1 += residual
        residual = x0
        x1 = self.conv_p2(x1).relu()
        x1 += residual
        residual = x0
        x1 = self.conv_p3(x1).relu()
        x0 += residual
        x1 = global_mean_pool(x1, data[1].batch)
        # x1 = torch.cat(x1, dim=0)

        x = torch.cat([x0, x1], dim=1)
        x = self.dense3(x).relu()
        x = self.dense4(x).relu()
        pred = self.dense5(x)

        return pred


class GCN(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(GCN, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)
        # NOTE: ADD TWO INDEPENDENT BRANCHES - NO WEIGHT SHARING
        # NOTE: ADD 1-D CONV ALONG BERT EMBEDDING DIM (768)
        self.dense1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense2 = GraphLinear(2*n_hidden, n_hidden)
        self.conv1 = GCNConv(n_hidden, n_hidden)
        # self.conv1 = GCN2Conv(n_hidden, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
        self.conv3 = GCNConv(n_hidden, n_hidden)
        self.dense3 = GraphLinear(2*n_hidden, 256)
        self.dense4 = GraphLinear(256, 128)
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
        pred = self.dense5(x)

        return pred
