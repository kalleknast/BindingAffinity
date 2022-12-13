import torch
from torch_geometric.nn import Linear as GraphLinear
from torch_geometric.nn import global_mean_pool, BatchNorm, GCNConv
from torch.nn import Embedding, Conv1d, Conv2d, Linear, Dropout
from torch.nn import AdaptiveMaxPool1d, AdaptiveMaxPool2d


class CNN2D1x1(torch.nn.Module):

    def __init__(self, dataset):
        super(CNN2D1x1, self).__init__()

        # Encode SMILES
        self.embedding_d = Embedding(num_embeddings=dataset.len_drug_vocab + 1,
                                     embedding_dim=128)
        self.conv2d_d1 = Conv2d(1, out_channels=32,
                                kernel_size=(4, 4), padding='valid')
        self.conv2d_d2 = Conv2d(32, out_channels=64, kernel_size=(4, 8),
                                stride=(1, 1), padding='valid')
        self.conv2d_d3 = Conv2d(64, out_channels=96, kernel_size=(4, 12),
                                stride=(1, 2), padding='valid')
        self.conv1x1_d = Conv2d(96, out_channels=10, kernel_size=(1, 1),
                                stride=(1, 3), padding='valid')
        self.max_pool_d = AdaptiveMaxPool2d((1, 18))

        # Encode protein
        self.embedding_p = Embedding(num_embeddings=dataset.len_prot_vocab + 1,
                                     embedding_dim=128)
        self.conv2d_p1 = Conv2d(1, out_channels=32, kernel_size=(8, 4),
                                stride=(2, 1), padding='valid')
        self.conv2d_p2 = Conv2d(32, out_channels=64, kernel_size=(12, 8),
                                stride=(3, 2), padding='valid')
        self.conv2d_p3 = Conv2d(64, out_channels=96, kernel_size=(12, 12),
                                stride=(3, 3), padding='valid')
        self.conv1x1_p = Conv2d(96, out_channels=10,
                                kernel_size=(1, 1), padding='valid')
        self.max_pool_p = AdaptiveMaxPool2d((1, 18))

        # Combined regressor
        self.dense_1 = Linear(360, 1024)
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
        xd = self.conv1x1_d(xd).relu()
        xd = self.max_pool_d(xd).squeeze()

        # Encode proteins
        xp = self.embedding_p(xp.int()).reshape((-1, 1, 1000, 128))
        xp = self.conv2d_p1(xp).relu()
        xp = self.conv2d_p2(xp).relu()
        xp = self.conv2d_p3(xp).relu()
        xp = self.conv1x1_p(xp).relu()
        xp = self.max_pool_p(xp).squeeze()

        # Common regression head
        x = torch.cat([xd.reshape((-1, 10 * 18)),
                       xp.reshape((-1, 10 * 18))], dim=1)
        x = self.dense_1(x).relu()
        x = self.dropout_1(x).relu()
        x = self.dense_2(x).relu()
        x = self.dropout_2(x).relu()
        x = self.dense_3(x).relu()
        pred = self.dense_4(x)

        return pred


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
        x = torch.cat([xd.reshape((-1, 96 * 119)),
                       xp.reshape((-1, 96 * 119))], dim=1)
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


class Hybrid(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(Hybrid, self).__init__()

        # GCN for drug encoding
        self.gconv_d1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.gconv_d2 = GCNConv(hidden_channels, hidden_channels)
        self.gconv_d3 = GCNConv(hidden_channels, 96)

        # Conv1D for protein encoding
        self.embedding_p = Embedding(num_embeddings=dataset.len_prot_vocab + 1,
                                     embedding_dim=128)
        self.conv1d_p1 = Conv1d(128, out_channels=32,
                                kernel_size=8, padding='valid')
        self.conv1d_p2 = Conv1d(32, out_channels=64,
                                kernel_size=8, padding='valid')
        self.conv1d_p3 = Conv1d(64, out_channels=96,
                                kernel_size=8, padding='valid')
        self.global_max_pool_p = AdaptiveMaxPool1d(1)

        # Common regression head
        self.dense_1 = Linear(96*2, 1024)
        self.dropout_1 = Dropout(p=0.1)
        self.dense_2 = Linear(1024, 1024)
        self.dropout_2 = Dropout(p=0.1)
        self.dense_3 = Linear(1024, 512)
        self.dense_4 = Linear(512, 1)
        torch.nn.init.normal_(self.dense_4.weight)

    def forward(self, xd, xp):

        # GCN for drug encoding
        xd, edge_index, batch = xd.x.to(torch.float32), xd.edge_index, xd.batch
        xd = self.gconv_d1(xd, edge_index).relu()
        xd = self.gconv_d2(xd, edge_index).relu()
        xd = self.gconv_d3(xd, edge_index).relu()
        xd = global_mean_pool(xd, batch)  # [batch_size, hidden_channels]

        # Conv1D for protein encoding - from DeepDTA
        xp = self.embedding_p(xp.int()).transpose(1, 2)
        xp = self.conv1d_p1(xp).relu()
        xp = self.conv1d_p2(xp).relu()
        xp = self.conv1d_p3(xp).relu()
        xp = self.global_max_pool_p(xp).squeeze()

        # Common regression head - from DeepDTA
        x = torch.cat([xd, xp], dim=1)
        x = self.dense_1(x).relu()
        x = self.dropout_1(x).relu()
        x = self.dense_2(x).relu()
        x = self.dropout_2(x).relu()
        x = self.dense_3(x).relu()
        pred = self.dense_4(x)

        return pred


class BertDTA(torch.nn.Module):

    def __init__(self, dataset):
        super(BertDTA, self).__init__()

        # Encode SMILES
        self.conv1d_d1 = Conv1d(1, out_channels=32,
                                kernel_size=4, padding='valid')
        self.conv1d_d2 = Conv1d(32, out_channels=64,
                                kernel_size=4, padding='valid')
        self.conv1d_d3 = Conv1d(64, out_channels=96,
                                kernel_size=4, padding='valid')
        self.global_max_pool_d = AdaptiveMaxPool1d(1)

        # Encode protein
        self.conv1d_p1 = Conv1d(1, out_channels=32,
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
        xd = global_mean_pool(xd.x, xd.batch).reshape(-1, 1, 768)
        xd = self.conv1d_d1(xd).relu()
        xd = self.conv1d_d2(xd).relu()
        xd = self.conv1d_d3(xd).relu()
        xd = self.global_max_pool_d(xd).squeeze()

        # Encode proteins
        xp = global_mean_pool(xp.x, xp.batch).reshape(-1, 1, 768)
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


class BertMLP(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(BertMLP, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)
        self.dense_d1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_d2 = GraphLinear(2*n_hidden, n_hidden)
        self.dense_p1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_p2 = GraphLinear(2*n_hidden, n_hidden)
        self.dense_c1 = GraphLinear(2*n_hidden, 256)
        self.dense_c2 = GraphLinear(256, 128)
        self.dense_c3 = GraphLinear(128, 1)

    def forward(self, xd, xp):
        # Drug
        # average pooling
        x0 = global_mean_pool(xd.x, xd.batch)
        x0 = self.batchnorm(x0)
        x0 = self.dense_d1(x0).relu()
        x0 = self.dense_d2(x0).relu()
        # x0 = torch.cat(x0, dim=0)

        # Protein
        # average pooling
        x1 = global_mean_pool(xp.x, xp.batch)
        x1 = self.batchnorm(x1)
        x1 = self.dense_p1(x1).relu()
        x1 = self.dense_p2(x1).relu()
        # x1 = torch.cat(x1, dim=0)

        x = torch.cat([x0, x1], dim=1)
        x = self.dense_c1(x).relu()
        x = self.dense_c2(x).relu()
        x = self.dense_c3(x)

        return x


class BertCNN(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(BertCNN, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)

        # Drug branch
        self.dense_d1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_d2 = GraphLinear(2*n_hidden, n_hidden)
        self.conv_d1 = Conv1d(1, 128, kernel_size=4)
        self.conv_d2 = Conv1d(128, 128, kernel_size=4)
        self.conv_d3 = Conv1d(128, 128, kernel_size=4)
        self.global_max_pool_d = AdaptiveMaxPool1d(1)

        # Protein branch
        self.dense_p1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_p2 = GraphLinear(2*n_hidden, n_hidden)
        self.conv_p1 = Conv1d(1, 128, kernel_size=4)
        self.conv_p2 = Conv1d(128, 128, kernel_size=4)
        self.conv_p3 = Conv1d(128, 128, kernel_size=4)
        self.global_max_pool_p = AdaptiveMaxPool1d(1)

        # Common regressor
        self.dense_1 = GraphLinear(2*n_hidden, 256)
        self.dense_2 = GraphLinear(256, 128)
        self.dense_3 = GraphLinear(128, 1)

    def forward(self, xd, xp):

        # Drug
        x0 = global_mean_pool(xd.x, xd.batch)
        x0 = self.batchnorm(x0)
        x0 = self.dense_d1(x0).relu()
        x0 = self.dense_d2(x0).relu().reshape(-1, 1, 128)
        # residual = x0
        x0 = self.conv_d1(x0).relu()
        # x0 += residual
        # residual = x0
        x0 = self.conv_d2(x0).relu()
        # x0 += residual
        # residual = x0
        x0 = self.conv_d3(x0).relu()
        x0 = self.global_max_pool_d(x0).squeeze()
        # x0 += residual
        # x0 = torch.cat(x0, dim=0)

        # Protein
        x1 = global_mean_pool(xp.x, xp.batch)
        x1 = self.batchnorm(x1)
        x1 = self.dense_p1(x1).relu()
        x1 = self.dense_p2(x1).relu().reshape(-1, 1, 128)
        # residual = x1
        x1 = self.conv_p1(x1).relu()
        # x1 += residual
        # residual = x0
        x1 = self.conv_p2(x1).relu()
        # x1 += residual
        # residual = x0
        x1 = self.conv_p3(x1).relu()
        x1 = self.global_max_pool_p(x1).squeeze()
        # x0 += residual
        # x1 = torch.cat(x1, dim=0)
        # import ipdb; ipdb.set_trace()
        x = torch.cat([x0, x1], dim=1)
        x = self.dense_1(x).relu()
        x = self.dense_2(x).relu()
        pred = self.dense_3(x)

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
