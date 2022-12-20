import torch
from torch_geometric.nn import Linear as GraphLinear
from torch_geometric.nn import global_mean_pool, BatchNorm, GCNConv
from torch.nn import Embedding, Conv1d, Conv2d, Linear, Dropout
from torch.nn import AdaptiveMaxPool1d, AdaptiveMaxPool2d
from transformers import AutoModel


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
    def __init__(self, dataset):
        super(Hybrid, self).__init__()

        # GCN for drug encoding
        self.gconv_d1 = GCNConv(dataset.num_node_features, 32)
        self.gconv_d2 = GCNConv(32, 64)
        self.gconv_d3 = GCNConv(64, 96)

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


class MTDTI(torch.nn.Module):

    def __init__(self, dataset):
        super(MTDTI, self).__init__()
        """
        From https://github.com/deargen/mt-dti
        https://github.com/deargen/mt-dti/blob/master/src/finetune/dti_model.py
        line 659:
            model_fn_v11(self, features, labels, mode, params)
        """

        # Encode SMILES
        # BERT
        # self.drug_encoder = AutoModel.from_pretrained(
        #     "seyonec/ChemBERTa_zinc250k_v2_40k")
        self.drug_encoder = AutoModel.from_pretrained(
            'DeepChem/ChemBERTa-77M-MTR')

        # Encode protein (DeepDTA-style)
        self.embedding_p = Embedding(num_embeddings=dataset.len_prot_vocab + 1,
                                     embedding_dim=128)
        self.conv1d_p1 = Conv1d(128, out_channels=32,
                                kernel_size=12, padding='valid')
        self.conv1d_p2 = Conv1d(32, out_channels=64,
                                kernel_size=12, padding='valid')
        self.conv1d_p3 = Conv1d(64, out_channels=96,
                                kernel_size=12, padding='valid')
        self.global_max_pool_p = AdaptiveMaxPool1d(1)

        # Combined regressor
        self.dense_1 = Linear(384 + 96, 1024)
        self.dropout_1 = Dropout(p=0.1)
        self.dense_2 = Linear(1024, 1024)
        self.dropout_2 = Dropout(p=0.1)
        self.dense_3 = Linear(1024, 512)
        self.dense_4 = Linear(512, 1)
        torch.nn.init.normal_(self.dense_4.weight)

    def forward(self, xd, xp):

        # Encode drugs/SMILES
        xd = self.drug_encoder(xd).pooler_output

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
        x0 = self.conv_d1(x0).relu()
        x0 = self.conv_d2(x0).relu()
        x0 = self.conv_d3(x0).relu()
        x0 = self.global_max_pool_d(x0).squeeze()

        # Protein
        x1 = global_mean_pool(xp.x, xp.batch)
        x1 = self.batchnorm(x1)
        x1 = self.dense_p1(x1).relu()
        x1 = self.dense_p2(x1).relu().reshape(-1, 1, 128)
        x1 = self.conv_p1(x1).relu()
        x1 = self.conv_p2(x1).relu()
        x1 = self.conv_p3(x1).relu()
        x1 = self.global_max_pool_p(x1).squeeze()

        x = torch.cat([x0, x1], dim=1)
        x = self.dense_1(x).relu()
        x = self.dense_2(x).relu()
        pred = self.dense_3(x)

        return pred


class BertGCN(torch.nn.Module):

    def __init__(self, dataset, n_hidden=128):
        super(BertGCN, self).__init__()

        self.batchnorm = BatchNorm(dataset.num_node_features)

        # Drug branch
        self.dense_d1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_d2 = GraphLinear(2*n_hidden, n_hidden)
        self.gconv_d1 = GCNConv(n_hidden, n_hidden)
        self.gconv_d2 = GCNConv(n_hidden, n_hidden)
        self.gconv_d3 = GCNConv(n_hidden, n_hidden)

        # Protein branch
        self.dense_p1 = GraphLinear(dataset.num_node_features, 2*n_hidden)
        self.dense_p2 = GraphLinear(2*n_hidden, n_hidden)
        self.gconv_p1 = GCNConv(n_hidden, n_hidden)
        self.gconv_p2 = GCNConv(n_hidden, n_hidden)
        self.gconv_p3 = GCNConv(n_hidden, n_hidden)

        # Common regressor
        self.dense_1 = GraphLinear(2*n_hidden, 256)
        self.dense_2 = GraphLinear(256, 128)
        self.dense_3 = Linear(128, 1)

    def forward(self, xd, xp):

        # Encode drugs/SMILES
        xd.x = self.batchnorm(xd.x)
        xd.x = self.dense_d1(xd.x).relu()
        xd.x = self.dense_d2(xd.x).relu()
        xd.x = self.gconv_d1(xd.x, xd.edge_index).relu()
        xd.x = self.gconv_d2(xd.x, xd.edge_index).relu()
        xd.x = self.gconv_d3(xd.x, xd.edge_index).relu()
        xd.x = global_mean_pool(xd.x, xd.batch)  # average pooling

        # Encode proteins
        xp.x = self.batchnorm(xp.x)
        xp.x = self.dense_p1(xp.x).relu()
        xp.x = self.dense_p2(xp.x).relu()
        xp.x = self.gconv_p1(xp.x, xp.edge_index).relu()
        xp.x = self.gconv_p2(xp.x, xp.edge_index).relu()
        xp.x = self.gconv_p3(xp.x, xp.edge_index).relu()
        xp.x = global_mean_pool(xp.x, xp.batch)  # average pooling

        # Common regression head
        x = torch.cat([xd.x, xp.x], dim=1)
        x = self.dense_1(x).relu()
        x = self.dense_2(x).relu()
        pred = self.dense_3(x)

        return pred
