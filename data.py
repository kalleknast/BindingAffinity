import random
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import pandas as pd
import json
from tdc.multi_pred import DTI
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


class DeepDTADatasetSimple(Dataset):
    def __init__(self, partition='train'):

        self.partition = partition
        train_df = pd.read_pickle('data/processed/train_data_fold1.pkl')
        valid_df = pd.read_pickle('data/processed/valid_data_fold1.pkl')
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


class DeepDTADataset(Dataset):
    def __init__(self, data_dir, dataset='KIBA', partition='train'):

        self.partition = partition
        self.raw_dir = data_dir
        self.dataset_name = dataset
        self.data = pd.read_csv(osp.join(data_dir,
                                f'DeepDTA_{dataset}.tsv'), sep='\t')
        self.y = np.stack(self.data['Y']).astype(np.int32).reshape((-1, 1))

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

        # Data split: 70/15/15
        self.drug_IDs = self.data['Drug_ID'].unique()
        self.n_drugs = self.drug_IDs.shape[0]
        # 70% to training
        self.n_train_drugs = int(round(self.n_drugs * 0.7))
        ids = list(range(self.n_drugs))
        np.random.shuffle(ids)
        self.train_drugs = self.drug_IDs[ids[:self.n_train_drugs]]
        self.train_ids = []
        for drug in self.train_drugs:
            self.train_ids += \
                list(self.data.index[self.data['Drug_ID'] == drug])
        # 15% to validation
        rem_drugs = np.setdiff1d(self.drug_IDs, self.train_drugs)
        self.n_valid_drugs = int(round(self.n_drugs * 0.15))
        ids = list(range(len(rem_drugs)))
        np.random.shuffle(ids)
        self.valid_drugs = rem_drugs[ids[:self.n_valid_drugs]]
        self.valid_ids = []
        for drug in self.valid_drugs:
            self.valid_ids += \
                list(self.data.index[self.data['Drug_ID'] == drug])
        # 15% to test (the rest)
        self.test_drugs = np.setdiff1d(rem_drugs, self.valid_drugs)
        self.n_test_drugs = self.test_drugs.shape[0]
        self.test_ids = []
        for drug in self.test_drugs:
            self.test_ids += \
                list(self.data.index[self.data['Drug_ID'] == drug])

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


class BindingAffinityDataset(GraphDataset):

    def __init__(self, root,
                 dataset_name='DAVIS',
                 partition='train',
                 network_type='gcn'):
        """
        """
        self.partition = partition
        self.dataset_name = dataset_name
        self.raw_file_name = f'{dataset_name}.tab'
        self.network_type = network_type

        super(BindingAffinityDataset, self).__init__(root, None, None)

    @property
    def raw_file_names(self):
        return self.raw_file_name

    @property
    def processed_file_names(self):

        prot_embed_fnames = [self._build_embed_fname(id) for id in self.prots]
        drug_embed_fnames = [self._build_embed_fname(id) for id in self.drugs]

        return prot_embed_fnames + drug_embed_fnames

    def download(self):

        raw = DTI(name=self.dataset_name, path=self.raw_dir)
        self.raw_data = raw.get_data()
        self.drugs = list(self.raw_data['Drug_ID'].unique())
        self.n_drug = len(self.drugs)
        self.prots = list(self.raw_data['Target_ID'].unique())
        self.n_prot = len(self.prots)
        self.n_total = self.n_drug + self.n_prot
        # Assume that drugs are novel and searched for while proteins are known
        # partition data on the drugs so that drugs in train
        # are not in valid or test, and drugs in valid are not in test
        n_train = int(round(self.n_drug * 0.7))
        n_valid = int(round(self.n_drug * 0.1))
        self.train_drugs = random.sample(self.drugs, n_train)
        not_train_drugs = list(np.setdiff1d(self.drugs, self.train_drugs))
        self.valid_drugs = random.sample(not_train_drugs, n_valid)
        self.test_drugs = list(np.setdiff1d(not_train_drugs, self.valid_drugs))

        self.train_ids = []
        for drug in self.train_drugs:
            self.train_ids += \
                list(self.raw_data.index[self.raw_data['Drug_ID'] == drug])

        self.valid_ids = []
        for drug in self.valid_drugs:
            self.valid_ids += \
                list(self.raw_data.index[self.raw_data['Drug_ID'] == drug])

        self.test_ids = []
        for drug in self.test_drugs:
            self.test_ids += \
                list(self.raw_data.index[self.raw_data['Drug_ID'] == drug])

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
            if row["Target_ID"] not in processed_prots:

                token_ids = torch.tensor(
                    [prot_tokenizer.encode(row['Target'])])
                embed = prot_model(token_ids)[0].squeeze()
                edges = edges_from_protein_seequence(row['Target'])
                data = {'embeddings': Data(x=embed, edge_index=edges),
                        'Target_ID': row['Target_ID']}
                fname = self._build_embed_fname(row["Target_ID"])
                torch.save(data, osp.join(self.processed_dir, fname))
                processed_prots.append(row['Target_ID'])

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
                                    self._build_embed_fname(row["Target_ID"]))
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
            # The distribution of the KIBA scores is depicted in the right
            # panel of Figure 1A. He et al. (2017) pre-processed
            # the KIBA scores as follows: (i) for each KIBA score, its negative
            # was taken, (ii) the minimum value among the negatives was chosen
            # and (iii) the absolute value of the minimum was added to all
            # negative scores, thus constructing the final form of
            # the KIBA scores.
            y = float(- row['Y'] + self.raw_data['Y'].max())

        meta = {'data_drug_id': str(drug_data['Drug_ID']),
                'data_prot_id': str(prot_data['Target_ID']),
                'raw_Drug_ID': str(row['Drug_ID']),
                'raw_Drug': row['Drug'],
                'raw_Target_ID': str(row['Target_ID']),
                'raw_Target': row['Target'],
                'raw_Y': row['Y']}

        return drug_data['embeddings'], prot_data['embeddings'], y, meta
