import torch
import sys
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
import json
# from tdc.multi_pred import DTI
from torch_geometric.utils import from_smiles


def train(data_loader, model, device, loss_fn, optimizer, verbose=False):

    model.train()
    data_loader.dataset.partition = 'train'
    ave_loss = 0.0

    if verbose:
        print('Batch loss:', end=' ')

    num_batch = len(data_loader)
    # Iterate in batches over the training dataset.
    # for batch, data in enumerate(dataloader):
    for batch, data in enumerate(data_loader):
        xd, xp = data[0].to(device), data[1].to(device)
        # Forward
        pred = model(xd, xp)
        # Compute the loss
        loss = loss_fn(pred, data[2].to(device, dtype=torch.float32))
        # Backpropagation
        loss.backward()
        # Weight update
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss.detach().item()
        ave_loss += batch_loss
        if verbose:
            s = f'{batch_loss:.3f}'
            print(s, end='\b'*len(s))
            sys.stdout.flush()

    if verbose:
        print()

    return ave_loss / num_batch


def evaluate(data_loader, model, device, loss_fn, partition='valid'):

    data_loader.dataset.partition = partition
    batch_size = data_loader.batch_size
    model.eval()
    num_batch = len(data_loader)
    ave_loss = 0.0
    preds = np.empty(batch_size * num_batch)
    targets = np.empty(batch_size * num_batch)

    with torch.no_grad():
        # Iterate in batches over the validation dataset.
        for batch, data in enumerate(data_loader):
            y = data[2].to(device, dtype=torch.float32)
            pred = model(data[0].to(device), data[1].to(device))
            ave_loss += loss_fn(pred, y).item()
            pred = pred.cpu().detach().numpy().flatten()
            i0, i1 = batch * batch_size, batch * batch_size + pred.shape[0]
            preds[i0: i1] = pred
            targets[i0: i1] = y.cpu().detach().numpy().flatten()

    return preds, targets, ave_loss / num_batch


def check_dataset(dataloader, epochs=2, compare='ID'):
    """
    Arguments
    ---------
    dataloader  : DataLoader instantiated with a dataset
    epochs      : The number of epochs to compare
    compare     : "ID" or "full"

    Return
    ------
    raw_data    : The loaded CSV/TSV file used to generate the dataset with
                  one dlY_{epoch} columns added for each epoch checked.
                  The dlY_{epoch} columns holds the Y values from the
                  dataloader and should be identical to the original Y values.
                  I.e. for the first epoch:
                    np.allclose(raw_data['Y'], raw_data['dlY_1'])
    """

    raw_data = dataloader.dataset.raw_data
    if dataloader.dataset.partition == 'train':
        raw_data = raw_data.loc[dataloader.dataset.train_ids]
    elif dataloader.dataset.partition == 'valid':
        raw_data = raw_data.loc[dataloader.dataset.valid_ids]
    elif dataloader.dataset.partition == 'test':
        raw_data = raw_data.loc[dataloader.dataset.test_ids]
    else:
        raise ValueError('Unknown dataset partition '
                         f'{dataloader.dataset.partition}.')

    for epoch in range(1, epochs+1):
        print('Epoch:', epoch)

        y_col = f'dlY_{epoch}'
        raw_data[y_col] = np.nan

        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            meta = data[-1]
            if compare == 'ID':
                drugs = meta['Drug_ID']
                prots = meta['Prot_ID']
            elif compare == 'full':
                drugs = meta['Drug']
                prots = meta['Prot']
            else:
                raise ValueError('Unknown comparision')

            for i, (drug, prot) in enumerate(zip(drugs, prots)):
                ids_d = raw_data['Drug_ID'] == drug
                ids_p = raw_data['Prot_ID'] == prot
                idx = np.logical_and(ids_d, ids_p)
                if idx.sum() != 1:
                    print(f'Problem with drug {drug} and protein {prot}.')
                    import ipdb
                    ipdb.set_trace()
                else:
                    raw_data.loc[idx, y_col] = float(data[2][i])

        print(f"Epoch  {epoch} all close: "
              f"{np.allclose(raw_data['Y'], raw_data[y_col])}")
    print()

    return raw_data


def get_node_edges(smiles_edges, index_map):
    """
    """
    node_edges = [[], []]
    for edge in smiles_edges.T:

        id_0 = np.logical_and(index_map['smiles_i0'] <= edge[0],
                              index_map['smiles_i1'] >= edge[0])
        id_1 = np.logical_and(index_map['smiles_i0'] <= edge[1],
                              index_map['smiles_i1'] >= edge[1])
        if id_0.sum() == 1 and id_1.sum() == 1:
            node_edges[0].append(int(index_map[id_0]['token_i']))
            node_edges[1].append(int(index_map[id_1]['token_i']))
        elif id_0.sum() > 1 or id_1.sum() > 1:
            raise ValueError('The edge seems to connect to multiple nodes!')

    return np.array(node_edges, dtype=int)


def smiles_edges_to_token_edges(smiles, tokenizer, reverse_vocab):
    """
    """
    token_ids = tokenizer.encode(smiles)
    index_map = get_indexmap(token_ids, reverse_vocab, smiles)
    smiles_edges = from_smiles(smiles).edge_index
    node_edges = get_node_edges(smiles_edges, index_map)
    # keep only between node edges
    node_edges = node_edges[:, ((node_edges[0] - node_edges[1]) != 0)]
    # remove duplicates. Duplicates can occur when different atoms within the
    # same nodes are connected to each other.
    node_edges = np.unique(node_edges, axis=1)

    return node_edges, index_map


def get_indexmap(token_ids, rev_vocab, smiles):

    index_map = pd.DataFrame(index=range(len(token_ids)),
                             columns=['token_i',
                                      'token',
                                      'token_id',
                                      'keep',
                                      'smiles_i0',
                                      'smiles_i1'])
    start = 0
    token_i = 0
    for i, token_id in enumerate(token_ids):

        token = rev_vocab[token_id]

        if token.isalpha():  # only all alphabetic chars are nodes
            smiles_i0 = smiles[start:].find(token)
            if smiles_i0 >= 0:
                smiles_i0 += start
                smiles_i1 = smiles_i0 + len(token)
                start = smiles_i1

                index_map.loc[i] = (token_i, token, token_id,
                                    True, smiles_i0, smiles_i1 - 1)
                token_i += 1
            else:
                raise ValueError('Node token not found in SMILES.\nCheck that '
                                 'token_ids are computed from smiles.')
        else:
            index_map.loc[i] = (-1, token, token_id, False, -1, -1)

    return index_map


def edges_from_protein_sequence(prot_seq):
    """
    Since we only have the primary protein sequence we only know of the peptide
    bonds between amino acids. I.e. only amino acids linked in the primary
    sequence will have edges between them.
    """
    n = len(prot_seq)
    # first and row in COO format
    # each node is connected to left and right except the first an last.
    edge_index = np.stack([np.repeat(np.arange(n), 2)[1:-1],
                           np.repeat(np.arange(n), 2)[1:-1]], axis=0)
    for i in range(0, n, 2):
        edge_index[1, i], edge_index[1, i+1] = \
            edge_index[1, i+1], edge_index[1, i]

    return torch.tensor(edge_index, dtype=torch.long)


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
