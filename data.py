import random
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import pandas as pd
import json
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Data
from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer
from utils import edges_from_protein_seequence
from utils import smiles_edges_to_token_edges, get_vocab


def get_dictionary(fname, raw_data, kind):
    """
    """
    if not osp.isfile(fname):
        entities = raw_data[kind].unique()
        vocab = list(set(entities.sum()))
        dictionary = {t: i+1 for i, t in enumerate(vocab)}
        with open(fname, 'w') as f:
            json.dump(dictionary, f)
    else:
        with open(fname, 'r') as f:
            dictionary = json.load(f)

    return dictionary


def encode_sequence(sequence, dictionary, encoded_len):
    """
    """
    enc = np.zeros(encoded_len, dtype=np.int32)
    for i, c in enumerate(sequence):
        if i == encoded_len - 1:
            break
        enc[i] = dictionary[c]

    return enc


def partition_data(data_splits, data):
    """
    Splits on the unique drugs,

    Assume that drugs are novel and searched for while proteins are known
    partition data on the drugs so that drugs in train are not in valid or
    test, and drugs in valid are not in test.
    """
    assert np.sum(data_splits) == 1., 'data_splits should sum to 1'

    drugs = list(data['Drug_ID'].unique())
    n_drug = len(drugs)
    n_train = int(round(n_drug * data_splits[0]))
    n_valid = int(round(n_drug * data_splits[1]))
    train = {'drugs': random.sample(drugs, n_train)}
    not_train_drugs = list(np.setdiff1d(drugs, train['drugs']))
    valid = {'drugs': random.sample(not_train_drugs, n_valid)}
    test = {'drugs': list(np.setdiff1d(not_train_drugs, valid['drugs']))}

    train['ids'] = []
    for drug in train['drugs']:
        train['ids'] += list(data.index[data['Drug_ID'] == drug])

    valid['ids'] = []
    for drug in valid['drugs']:
        valid['ids'] += list(data.index[data['Drug_ID'] == drug])

    test['ids'] = []
    for drug in test['drugs']:
        valid['ids'] += list(data.index[data['Drug_ID'] == drug])

    return train, valid, test, n_drug


class DeepDTADataset(Dataset):
    def __init__(self, partition='train', fold=1):
        """
        The dataset (with same folds) from https://github.com/hkmztrk/DeepDTA
        partition   : 'train' | 'valid' | 'test'
        fold        : (int) 1, 2, 3, 4 or 5
        """

        self.partition = partition
        train_df = pd.read_pickle(f'data/processed/train_data_fold{fold}.pkl')
        valid_df = pd.read_pickle(f'data/processed/valid_data_fold{fold}.pkl')
        self.train_x = [np.stack(train_df['Drug_enc']).astype(np.int32),
                        np.stack(train_df['Prot_enc']).astype(np.int32)]
        self.train_y = np.stack(train_df['Y']).reshape((-1, 1))
        self.valid_x = [np.stack(valid_df['Drug_enc']).astype(np.int32),
                        np.stack(valid_df['Prot_enc']).astype(np.int32)]
        self.valid_y = np.stack(valid_df['Y']).reshape((-1, 1))
        self.len_drug_vocab = 64
        self.len_prot_vocab = 25

    def __len__(self):
        if self.partition == 'train':
            return self.train_y.shape[0]
        elif self.partition == 'valid':
            return self.valid_y.shape[0]

    def __getitem__(self, idx):

        if self.partition == 'train':
            xd, xp = self.train_x[0][idx], self.train_x[1][idx]
            y = self.train_y[idx]
        elif self.partition == 'valid':
            xd, xp = self.valid_x[0][idx], self.valid_x[1][idx]
            y = self.valid_y[idx]

        return torch.tensor(xd), torch.tensor(xp), torch.tensor(y)


class EmbeddingDataset(Dataset):
    def __init__(self, data_dir, dataset='KIBA',
                 partition='train', data_splits=(.8, .2, 0.)):
        """
        The dataset is from https://github.com/hkmztrk/DeepDTA

        Arguments
        --------
        data_dir    : Where the dataset (e.g. raw/DeepDTA_KIBA.tsv) is stored.
        partition   : 'train' | 'valid' | 'test'.
        dataSet     : 'KIBA' | 'DAVIS' (only KIBA is currently available).
        data_splits : splits on the unique drugs, should sum to 1.
        """

        self.partition = partition
        self.raw_dir = data_dir
        self.dataset_name = dataset
        self.data_splits = data_splits
        self.data = pd.read_csv(osp.join(data_dir,
                                f'DeepDTA_{dataset}.tsv'), sep='\t')

        # Prepare or load the dictionaries for drugs and proteins
        # Drug dictionary
        fname = osp.join(self.raw_dir, f'drug_dict_{self.dataset_name}.json')
        self.drug_dict = get_dictionary(fname, self.data, 'Drug')
        self.len_drug_vocab = len(self.drug_dict)
        # Protein dictionary
        fname = osp.join(self.raw_dir, f'prot_dict_{self.dataset_name}.json')
        self.prot_dict = get_dictionary(fname, self.data, 'Protein')
        self.len_prot_vocab = len(self.prot_dict)

        if self.dataset_name == 'DAVIS':
            # DAVIS: 0-pad SMILES to 85 and proteins to 1200. Truncate above.
            self.drug_encoded_len = 85
            self.prot_encoded_len = 1200
        elif self.dataset_name == 'KIBA':
            # KIBA: 0-pad smiles to 100 and proteins to 1000. Truncate above.
            self.drug_encoded_len = 100
            self.prot_encoded_len = 1000

        # Encode the SMLILES and protein strings
        # Drugs/SMILES
        self.drugs = self.data['Drug'].unique()  # Unique drugs
        self.data['Drug_enc'] = [np.zeros(self.drug_encoded_len,
                                          dtype=np.int32)] * self.data.shape[0]
        for drug in self.drugs:
            enc = encode_sequence(drug, self.drug_dict, self.drug_encoded_len)
            self.data['Drug_enc'] = np.where(self.data['Drug'] == drug,
                                             pd.Series([enc]),
                                             self.data['Drug_enc'])
        # Proteins
        self.prots = self.data['Protein'].unique()  # Unique proteins
        self.data['Prot_enc'] = [np.zeros(self.prot_encoded_len,
                                          dtype=np.int32)] * self.data.shape[0]
        for prot in self.prots:
            enc = encode_sequence(prot, self.prot_dict, self.prot_encoded_len)
            self.data['Prot_enc'] = np.where(self.data['Protein'] == prot,
                                             pd.Series([enc]),
                                             self.data['Prot_enc'])

        # Data split
        train, valid, test, n_drug = partition_data(self.data_splits,
                                                    self.data)
        self.train_ids, self.train_drugs = train['ids'], train['drugs']
        self.valid_ids, self.valid_drugs = valid['ids'], valid['drugs']
        self.test_ids, self.test_drugs = test['ids'], test['drugs']

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_ids)
        elif self.partition == 'valid':
            return len(self.valid_ids)
        elif self.partition == 'test':
            return len(self.test_ids)

    def __getitem__(self, idx):

        if self.partition == 'train':
            idx = self.train_ids[idx]
        elif self.partition == 'valid':
            idx = self.valid_ids[idx]
        elif self.partition == 'test':
            idx = self.test_ids[idx]

        xd = self.data.loc[idx]['Drug_enc']
        xp = self.data.loc[idx]['Prot_enc']
        y = np.array([self.data.loc[idx]['Y']])

        return torch.tensor(xd), torch.tensor(xp), torch.tensor(y)


class BertDataset(Dataset):

    def __init__(self, data_dir,
                 dataset='KIBA',
                 partition='train',
                 data_splits=(.8, .2, 0.)):
        """
        """
        self.partition = partition
        self.dataset_name = dataset
        self.data = pd.read_csv(osp.join(data_dir, 'raw',
                                f'DeepDTA_{dataset}.tsv'), sep='\t')
        self.processed_dir = osp.join(data_dir, 'processed')
        self.data_splits = data_splits
        # Split data into train/valid/test sets
        train, valid, test, n_drug = partition_data(self.data_splits,
                                                    self.data)
        self.n_drug = n_drug
        self.train_ids, self.train_drugs = train['ids'], train['drugs']
        self.valid_ids, self.valid_drugs = valid['ids'], valid['drugs']
        self.test_ids, self.test_drugs = test['ids'], test['drugs']

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_ids)
        elif self.partition == 'valid':
            return len(self.valid_ids)
        elif self.partition == 'test':
            return len(self.test_ids)

    def _build_embed_fname(self, ID):
        return f'{self.dataset_name}_{ID}_embedded.pt'

    def __getitem__(self, idx):

        if self.partition == 'train':
            row = self.data.loc[self.train_ids[idx]]
        elif self.partition == 'valid':
            row = self.data.loc[self.valid_ids[idx]]
        elif self.partition == 'test':
            row = self.data.loc[self.test_ids[idx]]
        else:
            row = self.data.loc[self.raw_data.index[idx]]

        prot_embed_fname = osp.join(self.processed_dir,
                                    self._build_embed_fname(row["Prot_ID"]))
        prot_data = torch.load(prot_embed_fname)

        drug_embed_fname = osp.join(self.processed_dir,
                                    self._build_embed_fname(row["Drug_ID"]))
        drug_data = torch.load(drug_embed_fname)

        if self.dataset_name == 'DAVIS':
            # see https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md
            y = float(- np.log(row['Y'] / 1e9))
        elif self.dataset_name == 'KIBA':
            y = row['Y']

        # meta = {'data_drug_id': str(drug_data['Drug_ID']),
        #         'data_prot_id': str(prot_data['Prot_ID']),
        #         'raw_Drug_ID': str(row['Drug_ID']),
        #         'raw_Drug': row['Drug'],
        #         'raw_Target_ID': str(row['Prot_ID']),
        #         'raw_Target': row['Protein'],
        #         'raw_Y': row['Y']}
        import ipdb; ipdb.set_trace()
        return drug_data['embeddings'].x, prot_data['embeddings'].x, y


class BertGraphDataset(GraphDataset):

    def __init__(self, root,
                 dataset_name='KIBA',
                 partition='train',
                 network_type='gcn',
                 data_splits=(.8, .2, 0.)):
        """
        """
        self.partition = partition
        self.dataset_name = dataset_name
        self.raw_file_name = f'DeepDTA_{dataset_name}.tsv'
        self.network_type = network_type
        self.data_splits = data_splits

        super(BertGraphDataset, self).__init__(root, None, None)

        # Split data into train/valid/test sets
        train, valid, test, n_drug = partition_data(self.data_splits,
                                                    self.raw_data)
        self.n_drug = n_drug
        self.train_ids, self.train_drugs = train['ids'], train['drugs']
        self.valid_ids, self.valid_drugs = valid['ids'], valid['drugs']
        self.test_ids, self.test_drugs = test['ids'], test['drugs']

    @property
    def raw_file_names(self):
        return self.raw_file_name

    @property
    def processed_file_names(self):
        self.raw_data = pd.read_csv(osp.join(self.raw_dir, self.raw_file_name),
                                    sep='\t')
        self.prots = self.raw_data['Prot_ID'].unique()
        self.drugs = self.raw_data['Drug_ID'].unique()
        self.n_prot, self.n_drug = len(self.prots), len(self.drugs)
        self.n_total = self.n_prot + self.n_drug
        prot_embed_fnames = [self._build_embed_fname(id) for id in self.prots]
        drug_embed_fnames = [self._build_embed_fname(id) for id in self.drugs]

        return prot_embed_fnames + drug_embed_fnames

    def download(self):
        pass

    def _build_embed_fname(self, ID):
        return f'{self.dataset_name}_{ID}_embedded.pt'

    def process(self):

        prot_model = ProteinBertModel.from_pretrained('bert-base')
        prot_tokenizer = TAPETokenizer(vocab='iupac')
        drug_model = AutoModel.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k")
        drug_tokenizer = AutoTokenizer.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k")
        vocab, reverse_vocab = get_vocab()

        processed_prots, processed_drugs = [], []

        for idx, row in tqdm(self.raw_data.iterrows(), total=self.n_total):

            # Only embed a protein if it hasn't already been embedded and saved
            if row["Prot_ID"] not in processed_prots:

                token_ids = torch.tensor(
                    [prot_tokenizer.encode(row['Protein'])])
                embed = prot_model(token_ids)[0].squeeze()
                edges = edges_from_protein_seequence(row['Protein'])
                data = {'embeddings': Data(x=embed, edge_index=edges),
                        'Prot_ID': row['Prot_ID']}
                fname = self._build_embed_fname(row["Prot_ID"])
                torch.save(data, osp.join(self.processed_dir, fname))
                processed_prots.append(row['Prot_ID'])

            # Only embed a drug if it hasn't already been embedded and saved
            if row['Drug_ID'] not in processed_drugs:

                token_ids = torch.tensor([drug_tokenizer.encode(row['Drug'])])
                embed = drug_model(token_ids).last_hidden_state.squeeze()
                edges, index_map = smiles_edges_to_token_edges(row['Drug'],
                                                               drug_tokenizer,
                                                               reverse_vocab)
                data = {'embeddings': Data(x=embed,
                                           edge_index=torch.tensor(edges)),
                        'Drug_ID': row['Drug_ID'],
                        'node_ids': index_map['keep'].values.astype('bool')}
                fname = self._build_embed_fname(row["Drug_ID"])
                torch.save(data, osp.join(self.processed_dir, fname))
                processed_drugs.append(row['Drug_ID'])

    def len(self):

        if self.partition == 'train':
            n = len(self.train_ids)
        elif self.partition == 'valid':
            n = len(self.valid_ids)
        elif self.partition == 'test':
            n = len(self.test_ids)
        else:
            n = len(self.raw_data.index)

        return n

    def get(self, idx):

        if self.partition == 'train':
            row = self.raw_data.loc[self.train_ids[idx]]
        elif self.partition == 'valid':
            row = self.raw_data.loc[self.valid_ids[idx]]
        elif self.partition == 'test':
            row = self.raw_data.loc[self.test_ids[idx]]
        else:
            row = self.raw_data.loc[self.raw_data.index[idx]]

        prot_embed_fname = osp.join(self.processed_dir,
                                    self._build_embed_fname(row["Prot_ID"]))
        prot_data = torch.load(prot_embed_fname)

        drug_embed_fname = osp.join(self.processed_dir,
                                    self._build_embed_fname(row["Drug_ID"]))
        drug_data = torch.load(drug_embed_fname)

        if self.network_type == 'gcn':  # remove non-node embeddings
            node_ids = drug_data['node_ids']
            drug_data['embeddings'].x = drug_data['embeddings'].x[node_ids]
        elif self.network_type == 'mlp':  # keep both bond/edge and node embeds
            pass
        else:
            raise ValueError('Unknown network type')

        if self.dataset_name == 'DAVIS':
            # see https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md
            y = float(- np.log(row['Y'] / 1e9))
        elif self.dataset_name == 'KIBA':
            y = row['Y']

        meta = {'data_drug_id': str(drug_data['Drug_ID']),
                'data_prot_id': str(prot_data['Prot_ID']),
                'raw_Drug_ID': str(row['Drug_ID']),
                'raw_Drug': row['Drug'],
                'raw_Target_ID': str(row['Prot_ID']),
                'raw_Target': row['Protein'],
                'raw_Y': row['Y']}

        return drug_data['embeddings'], prot_data['embeddings'], y
