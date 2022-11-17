import os.path as osp
from glob import glob
import torch
from torch_geometric.data import Dataset
from tdc.multi_pred import DTI
from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import pandas as pd


def edges_from_protein_seequence(prot_seq):
    """
    Since we only have the primary protein sequence we only know of the peptide
    bonds between amino acids. I.e. only amino acids linked in the primary
    sequence will have edges between them.
    """
    n = len(prot_seq)
    # first row in COO format
    # each node is connected to left and right except the first an last.
    row0 = np.repeat(np.arange(n), 2)[1:-1]
    # second row in COO format
    row1 = row0.copy()
    for i in range(0, len(row0), 2):
        row1[i], row1[i+1] = row1[i+1], row1[i]

    edge_index = torch.tensor([row0, row1], dtype=torch.long)

    return edge_index


class BindingAffinityDataset(Dataset):
    def __init__(self, root, dataset_name='DAVIS', test=False):
        """
        """
        self.test = test
        self.dataset_name = dataset_name
        self.raw_file_name = f'{dataset_name}.tab'

        super(BindingAffinityDataset, self).__init__(root, None, None)

    @property
    def raw_file_names(self):
        return self.raw_file_name

    @property
    def processed_file_names(self):

        if self.test:
            self.master = pd.read_csv(
                osp.join(self.processed_dir,
                         (f'master_test_{self.dataset_name}.csv')))

        else:
            self.master = pd.read_csv(
                osp.join(self.processed_dir,
                         (f'master_test_{self.dataset_name}.csv')))

        prot_ids = self.master['Target_ID'].unique()
        prot_embed_fnames = [f'{id}_embedded.pt' for id in prot_ids]
        drug_ids = self.master['Drug_ID'].unique()
        drug_embed_fnames = [f'{id}_embedded.pt' for id in drug_ids]
        filenames = prot_embed_fnames + drug_embed_fnames

        return filenames

    def download(self):

        raw_data = DTI(name=self.dataset_name, path=self.raw_dir)
        split = raw_data.get_split()

        self.raw_train = split['train']
        self.raw_train['prot_embed_fname'] = ' ' * 40
        self.raw_train['drug_embed_fname'] = ' ' * 93
        self.raw_train['log_y'] = np.nan
        self.raw_train.to_csv(osp.join(self.processed_dir,
                              f'master_{self.dataset_name}.csv'),
                              index=False)
        self.raw_test = split['test']
        self.raw_test['prot_embed_fname'] = ' ' * 40
        self.raw_test['drug_embed_fname'] = ' ' * 93
        self.raw_test['log_y'] = np.nan
        self.raw_train.to_csv(osp.join(self.processed_dir,
                              f'master_test_{self.dataset_name}.csv'),
                              index=False)

    def process(self):

        prot_model = ProteinBertModel.from_pretrained('bert-base')
        prot_tokenizer = TAPETokenizer(vocab='iupac')
        # drug_model = AutoModelWithLMHead.from_pretrained(
        #    "seyonec/ChemBERTa_zinc250k_v2_40k")
        drug_model = AutoModel.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k")
        drug_tokenizer = AutoTokenizer.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k")

        if self.test:
            raw = self.raw_test
        else:
            raw = self.raw_train

        for idx, row in tqdm(raw.iterrows(), total=raw.shape[0]):

            raw['log_y'].loc[idx] = np.log(row['Y'])

            embed_fname = f'{row["Target_ID"]}_embedded.pt'
            # Only embed a protein if it hasn't already been embedded and saved
            if embed_fname not in raw['prot_embed_fname']:

                token_ids = torch.tensor(
                    [prot_tokenizer.encode(row['Target'])])
                embed = prot_model(token_ids)[0].squeeze()
                edges = edges_from_protein_seequence(row['Target'])
                fname = osp.join(self.processed_dir, embed_fname)
                data = {'embeddings': Data(x=embed, edge_index=edges),
                        'Target_ID': row['Target_ID']}
                torch.save(data, fname)
            # Add the embed filename to the master file
            raw['prot_embed_fname'].loc[idx] = embed_fname

            embed_fname = f'{row["Drug_ID"]}_embedded.pt'
            # Only embed a drug if it hasn't already been embedded and saved
            if embed_fname not in raw['drug_embed_fname']:

                token_ids = torch.tensor([drug_tokenizer.encode(row['Drug'])])
                embed = drug_model(token_ids).last_hidden_state.squeeze()
                edges = from_smiles(row['Drug']).edge_index
                fname = osp.join(self.processed_dir, embed_fname)
                data = {'embeddings': Data(x=embed, edge_index=edges),
                        'Drug_ID': row['Drug_ID']}
                torch.save(data, fname)
            # Add the embed filename to the master file
            raw['drug_embed_fname'].loc[idx] = embed_fname

        if self.test:
            fname = osp.join(self.processed_dir,
                             (f'master_test_{self.dataset_name}.csv'))
        else:
            fname = osp.join(self.processed_dir,
                             f'master_{self.dataset_name}.csv')

        raw.to_csv(fname, index=False)

    def len(self):

        return len(self.processed_file_names)

    def get(self, idx):

        filename = self.processed_file_names[idx]
        data = torch.load(filename)

        return [data['drug'], data['protein']], data['log_y']
