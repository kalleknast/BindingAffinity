import random
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from tdc.multi_pred import DTI
from torch_geometric.data import Dataset, Data
from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer
from utils import edges_from_protein_seequence
from utils import smiles_edges_to_token_edges, get_vocab


class BindingAffinityDataset(Dataset):

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
        return f'{ID}_embedded.pt'

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
                # embed = embed[index_map['keep']]
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

        log_y = np.log(row['Y'])

        return drug_data['embeddings'], prot_data['embeddings'], log_y
