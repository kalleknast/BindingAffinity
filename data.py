import random
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import pandas as pd
from torch_geometric.utils import from_smiles
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GraphDataset, Data
from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer
from utils import edges_from_protein_sequence, smiles_edges_to_token_edges
from utils import get_dictionary

DRUG_BERT_NAME = 'DeepChem/ChemBERTa-77M-MTR'
# OLD:  DRUG_BERT_NAME = "seyonec/ChemBERTa_zinc250k_v2_40k"
PROT_BERT_NAME = 'bert-base'

# Drug dictionary from DeepDTA
DTA_DRUG_DICT = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34,
                 ".": 2, "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5,
                 "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8,
                 "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45,
                 "H": 12, "K": 46, "M": 47, "L": 13, "O": 48, "N": 14, "P": 15,
                 "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52,
                 "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24,
                 "m": 60, "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63,
                 "t": 28, "y": 64}
# Protein dictionary from DeepDTA
DTA_PROT_DICT = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
                 "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
                 "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
                 "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

drug_tokenizer = AutoTokenizer.from_pretrained(DRUG_BERT_NAME)
prot_tokenizer = TAPETokenizer(vocab='iupac')


def add_token_column(data, kind, tokenizer, mol_dict, max_len):
    """
    In-place

    Arguments:
    data        : A pandas DataFrame
    kind        : Drug or protein or tokens. I.e. tokenize drugs or proteins.
                  "prot" or "drug"
    mol_dict    : A dictionary mapping from vocab to tokens. Only important
                  when tokenizer is 'DeepDTA'. mol_dict is in-built with the
                  BERT tokenizers ('BERT-drug' or 'BERT_prot').
    tokenizer   : 'DeepDTA', or "BERT-drug" or 'BERT-prot'.
    max_len     : Length of the tokenized sequence. Only relevant when using
                  tokenizer == 'DeepDTA'.
    """
    token_col = f'{kind[:4]}_enc'
    mols = data[kind].unique()  # Unique drugs or prot
    data[token_col] = [np.zeros(max_len, dtype=np.int32)] * data.shape[0]
    for mol in mols:
        if tokenizer == 'DeepDTA':
            enc = tokenize_sequence(mol, mol_dict, max_len)
        elif tokenizer == 'BERT-drug':
            """
            Max drug (input) length for the drug_model (BERT):
                drug_model.config["max_position_embeddings"]: 514
            Max drug length in Kiba is 502.
            Padding lenght for tokenizer: 512
            """
            enc = drug_tokenizer.encode(mol, padding="max_length",
                                        truncation=True, max_length=512)
        elif tokenizer == 'BERT-prot':
            """
            Max protein (input) length for the prot_model (BERT):
                prot_model.config["max_position_embeddings"]: 8192
            Max protein length in Kiba is 4130.
            Padding lenght for tokenizer: 4200
            """
            enc = prot_tokenizer.encode(mol, padding="max_length",
                                        max_length=4200)

        data[token_col] = np.where(data[kind] == mol,
                                   pd.Series([enc]), data[token_col])


def tokenize_sequence(sequence, dictionary, max_len):
    """
    """
    enc = np.zeros(max_len, dtype=np.int32)
    max_len = min(max_len, len(sequence))

    for i, c in enumerate(sequence[:max_len]):
        enc[i] = dictionary[c]

    return enc


def partition_data(data_splits, data, kind='drug'):
    """
    data_splits : data_splits should sum to 1
    data        :
    kind        : "pair" (splits on pairs, DeepDTA-style) or
                  "drugs" (splits on the unique drugs)

    Assume that drugs are novel and searched for while proteins are known
    partition data on the drugs so that drugs in train are not in valid or
    test, and drugs in valid are not in test.
    """
    assert np.sum(data_splits) == 1., 'data_splits should sum to 1'

    drugs = list(data['Drug_ID'].unique())
    n_drug = len(drugs)

    if kind == 'drug':
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
            test['ids'] += list(data.index[data['Drug_ID'] == drug])

    elif kind == 'pair':
        n = len(data)
        n_train = int(round(n * data_splits[0]))
        n_valid = int(round(n * data_splits[1]))
        n_test = int(round(n * data_splits[2]))
        ids = np.arange(n, dtype=int)
        random.shuffle(ids)
        train = {'ids': ids[:n_train]}
        train['drugs'] = data.loc[train['ids'], 'Drug_ID'].unique()
        valid = {'ids': ids[n_train:n_train+n_valid]}
        valid['drugs'] = data.loc[valid['ids'], 'Drug_ID'].unique()
        test = {'ids': ids[n_train+n_valid:]}
        test['drugs'] = data.loc[test['ids'], 'Drug_ID'].unique()

    return train, valid, test, n_drug


def partition_data_DTA(data):
    """
    """
    drugs = list(data['Drug_ID'].unique())
    n_drug = len(drugs)

    n_train = 78836
    n_valid = 19709

    train = {'ids': np.arange(n_train, dtype=int)}
    valid = {'ids': np.arange(n_valid, n_train+n_valid, dtype=int)}
    test = {'ids': np.arange(0, dtype=int)}

    return train, valid, test, n_drug


class DeepDTADataset(Dataset):
    def __init__(self, partition='train', fold=1):
        """
        The dataset (with same folds) from https://github.com/hkmztrk/DeepDTA
        partition   : 'train' | 'valid'
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
            xd = torch.tensor(self.train_x[0][idx])
            xp = torch.tensor(self.train_x[1][idx])
            y = torch.tensor(self.train_y[idx], dtype=torch.float32)
        elif self.partition == 'valid':
            xd = torch.tensor(self.valid_x[0][idx])
            xp = torch.tensor(self.valid_x[1][idx])
            y = torch.tensor(self.valid_y[idx], dtype=torch.float32)

        return xd, xp, y


class EmbeddingDataset(Dataset):
    def __init__(self, root, dataset_name='KIBA', partition='train',
                 data_splits=(.8, .2, 0.), partition_kind='drug',
                 drug_tokenizer='DeepDTA', prot_tokenizer='DeepDTA'):
        """
        The raw Kiba dataset is from https://github.com/hkmztrk/DeepDTA

        Arguments
        --------
        root            : Root dir, the dataset shuld be stored in root/raw
                          E.g. root/raw/DeepDTA_KIBA.tsv) is stored.
        partition       : 'train' | 'valid' | 'test'.
        dataset_name    : 'KIBA' | 'DAVIS' (only KIBA is currently available).
        data_splits     : splits on the unique drugs, should sum to 1.
        partition_kind  : "pair" or "drug". How to split the data.
                          "pair" - splits on drug-protein pairs
                          "drug" - splits on the unique drugs
        drug_tokenizer    : "DeepDTA" or "BERT-drug"
        prot_tokenizer    : "DeepDTA" or "BERT-prot"
        """

        self.partition = partition
        self.raw_dir = osp.join(root, 'raw')
        self.dataset_name = dataset_name
        self.data_splits = data_splits
        self.partition_kind = partition_kind
        self.drug_tokenizer = drug_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.raw_data = pd.read_csv(osp.join(self.raw_dir,
                                    f'DeepDTA_{dataset_name}.tsv'), sep='\t')

        # Prepare or load the dictionaries for drugs and proteins
        # Drug dictionary
        fname = osp.join(self.raw_dir, f'drug_dict_{self.dataset_name}.json')
        self.drug_dict = get_dictionary(fname, self.raw_data, 'Drug')
        # self.drug_dict = DTA_DRUG_DICT
        self.len_drug_vocab = len(self.drug_dict)
        # Protein dictionary
        fname = osp.join(self.raw_dir, f'prot_dict_{self.dataset_name}.json')
        self.prot_dict = get_dictionary(fname, self.raw_data, 'Protein')
        # self.prot_dict = DTA_PROT_DICT
        self.len_prot_vocab = len(self.prot_dict)

        if self.dataset_name == 'DAVIS':
            # DAVIS: 0-pad SMILES to 85 and proteins to 1200. Truncate above.
            self.drug_tokenized_len = 85
            self.prot_tokenized_len = 1200
        elif self.dataset_name == 'KIBA':
            # KIBA: 0-pad smiles to 100 and proteins to 1000. Truncate above.
            self.drug_tokenized_len = 100
            self.prot_tokenized_len = 1000

        # Encode the SMLILES and protein strings
        # Drugs/SMILES
        add_token_column(self.raw_data, 'Drug', self.drug_tokenizer,
                         self.drug_dict, self.drug_tokenized_len)
        # Proteins
        add_token_column(self.raw_data, 'Protein', self.prot_tokenizer,
                         self.prot_dict, self.prot_tokenized_len)

        # Data split
        # train, valid, test, n_drug = partition_data_DTA(self.raw_data)
        # self.train_ids = train['ids']
        # self.valid_ids = valid['ids']
        # self.test_ids = test['ids']
        train, valid, test, n_drug = partition_data(self.data_splits,
                                                    self.raw_data,
                                                    kind=self.partition_kind)
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
            row = self.raw_data.loc[self.train_ids[idx]]
        elif self.partition == 'valid':
            row = self.raw_data.loc[self.valid_ids[idx]]
        elif self.partition == 'test':
            row = self.raw_data.loc[self.test_ids[idx]]

        xd = torch.tensor(row['Drug_enc'])
        xp = torch.tensor(row['Prot_enc'])
        y = torch.tensor([row['Y']], dtype=torch.float32)
        meta = {'Drug_ID': str(row['Drug_ID']),
                'Drug': row['Drug'],
                'Prot_ID': str(row['Prot_ID']),
                'Prot': row['Protein'],
                'Y': row['Y']}

        return xd, xp, y, meta


class HybridDataset(GraphDataset):

    def __init__(self, root, dataset_name='KIBA', partition='train',
                 data_splits=(.8, .2, 0.), partition_kind='drug'):
        """
        partition_kind  : "pair" or "drug". How to split the data.
                          "pair" - splits on drug-protein pairs
                          "drug" - splits on the unique drugs
        """
        self.partition = partition
        self.dataset_name = dataset_name
        self.partition_kind = partition_kind
        self.raw_file_name = f'DeepDTA_{dataset_name}.tsv'
        self.raw_data = pd.read_csv(osp.join(root, 'raw',
                                    self.raw_file_name), sep='\t')
        self.data_splits = data_splits
        self.drug_tokenizer = 'DeepDTA'
        self.prot_tokenizer = 'DeepDTA'

        super(HybridDataset, self).__init__(root, None, None)

        # Prepare or load the dictionaries for drugs and proteins
        # Drug dictionary
        fname = osp.join(self.raw_dir, f'drug_dict_{self.dataset_name}.json')
        self.drug_dict = get_dictionary(fname, self.raw_data, 'Drug')
        self.len_drug_vocab = len(self.drug_dict)
        # Protein dictionary
        fname = osp.join(self.raw_dir, f'prot_dict_{self.dataset_name}.json')
        self.prot_dict = get_dictionary(fname, self.raw_data, 'Protein')
        self.len_prot_vocab = len(self.prot_dict)

        if self.dataset_name == 'DAVIS':
            # DAVIS: 0-pad SMILES to 85 and proteins to 1200. Truncate above.
            self.drug_tokenized_len = 85
            self.prot_tokenized_len = 1200
        elif self.dataset_name == 'KIBA':
            # KIBA: 0-pad smiles to 100 and proteins to 1000. Truncate above.
            self.drug_tokenized_len = 100
            self.prot_tokenized_len = 1000

        # Encode the SMLILES and protein strings
        # Drugs/SMILES
        add_token_column(self.raw_data, 'Drug', self.drug_tokenizer,
                         self.drug_dict, self.drug_tokenized_len)
        # Proteins
        add_token_column(self.raw_data, 'Protein', self.prot_tokenizer,
                         self.prot_dict, self.prot_tokenized_len)

        # Split data into train/valid/test sets
        train, valid, test, n_drug = partition_data(self.data_splits,
                                                    self.raw_data,
                                                    kind=self.partition_kind)
        self.n_drug = n_drug
        self.train_ids, self.train_drugs = train['ids'], train['drugs']
        self.valid_ids, self.valid_drugs = valid['ids'], valid['drugs']
        self.test_ids, self.test_drugs = test['ids'], test['drugs']

    @property
    def raw_file_names(self):
        return self.raw_file_name

    @property
    def processed_file_names(self):

        return self.raw_file_name

    def download(self):
        pass

    def process(self):
        pass

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

        if self.dataset_name == 'DAVIS':
            # see https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md
            y = torch.tensor([- np.log(row['Y'] / 1e9)], dtype=torch.float32)
        elif self.dataset_name == 'KIBA':
            y = torch.tensor([row['Y']], dtype=torch.float32)

        meta = {'Drug_ID': str(row['Drug_ID']),
                'Drug': row['Drug'],
                'Prot_ID': str(row['Prot_ID']),
                'Prot': row['Protein'],
                'Y': row['Y']}

        # drug_data = torch.tensor(row['Drug_enc'])
        prot_data = torch.tensor(row['Prot_enc'])
        drug_data = from_smiles(row['Drug'])

        return drug_data, prot_data, y, meta


class BertDataset(GraphDataset):

    def __init__(self, root, dataset_name='KIBA', partition='train',
                 data_splits=(.8, .2, 0.), partition_kind='drug'):
        """
        partition_kind  : "pair" or "drug". How to split the data.
                          "pair" - splits on drug-protein pairs
                          "drug" - splits on the unique drugs
        """
        self.partition = partition
        self.partition_kind = partition_kind
        self.dataset_name = dataset_name
        self.raw_file_name = f'DeepDTA_{dataset_name}.tsv'
        self.data_splits = data_splits

        super(BertDataset, self).__init__(root, None, None)

        # Split data into train/valid/test sets
        train, valid, test, n_drug = partition_data(self.data_splits,
                                                    self.raw_data,
                                                    kind=self.partition_kind)
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

        prot_model = ProteinBertModel.from_pretrained(PROT_BERT_NAME)
        prot_tokenizer = TAPETokenizer(vocab='iupac')
        drug_model = AutoModel.from_pretrained(DRUG_BERT_NAME)
        drug_tokenizer = AutoTokenizer.from_pretrained(DRUG_BERT_NAME)
        vocab = drug_tokenizer.vocab
        reverse_vocab = {key: val for val, key in vocab.items()}

        processed_prots, processed_drugs = [], []

        for idx, row in tqdm(self.raw_data.iterrows(), total=self.n_total):

            # Only embed a protein if it hasn't already been embedded and saved
            if row["Prot_ID"] not in processed_prots:

                token_ids = torch.tensor(
                    [prot_tokenizer.encode(row['Protein'])])
                embed = prot_model(token_ids)[0].squeeze()
                edges = edges_from_protein_sequence(row['Protein'])
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
                data = {'embeddings':
                        Data(x=embed,
                             edge_index=torch.tensor(edges, dtype=torch.long)),
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

        if self.dataset_name == 'DAVIS':
            # see https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md
            y = torch.tensor([- np.log(row['Y'] / 1e9)], dtype=torch.float32)
        elif self.dataset_name == 'KIBA':
            y = torch.tensor([row['Y']], dtype=torch.float32)

        meta = {'Drug_ID': str(drug_data['Drug_ID']),
                'Prot_ID': str(prot_data['Prot_ID']),
                'raw_Drug_ID': str(row['Drug_ID']),
                'Drug': row['Drug'],
                'raw_Prot_ID': str(row['Prot_ID']),
                'Prot': row['Protein'],
                'Y': row['Y']}

        # drug_data['embeddings'].x.requires_grad = False
        # prot_data['embeddings'].x.requires_grad = False

        return drug_data['embeddings'], prot_data['embeddings'], y, meta
