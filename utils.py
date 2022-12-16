import torch
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from tdc.multi_pred import DTI
from transformers import AutoTokenizer
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


def find_edge_mismatches(filenames):
    """
    """
    raw = DTI(name='DAVIS').get_data()
    tokenizer = AutoTokenizer.from_pretrained(
        "seyonec/ChemBERTa_zinc250k_v2_40k")

    mismatches = []

    for filename in filenames:
        data = torch.load(filename)
        num_nodes = data['embeddings'].x.shape[0]
        last_node = int(data['embeddings'].edge_index.max())

        if num_nodes < (last_node + 1):
            smiles = np.unique(raw[data['Drug_ID'] == raw['Drug_ID']]['Drug'])
            mismatch = {'Drug_ID': data['Drug_ID'],
                        'SMILES': smiles[0],
                        'tokens': tokenizer(smiles[0])['input_ids'],
                        'diff': (last_node + 1) - num_nodes,
                        'nodes': data['embeddings'].x,
                        'edges': data['embeddings'].edge_index}
            mismatches.append(mismatch)

    return mismatches


bad_tokens = ["NCc", "CCCCC", "CNc", "Clc", "OCCOC"]


def get_vocab():
    """
    Tokenizer vocabulary for the drug tokenizer
    (seyonec/ChemBERTa_zinc250k_v2_40k).

    From https://huggingface.co/seyonec/ChemBERTa_zinc250k_v2_40k/blob/main/vocab.json
    """
    vocab = {
        "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "!": 5,
        "\"": 6, "#": 7, "$": 8, "%": 9, "&": 10, "'": 11, "(": 12, ")": 13,
        "*": 14, "+": 15, ",": 16, "-": 17, ".": 18, "/": 19, "0": 20, "1": 21,
        "2": 22, "3": 23, "4": 24, "5": 25, "6": 26, "7": 27, "8": 28, "9": 29,
        ":": 30, ";": 31, "<": 32, "=": 33, ">": 34, "?": 35, "@": 36, "A": 37,
        "B": 38, "C": 39, "D": 40, "E": 41, "F": 42, "G": 43, "H": 44, "I": 45,
        "J": 46, "K": 47, "L": 48, "M": 49, "N": 50, "O": 51, "P": 52, "Q": 53,
        "R": 54, "S": 55, "T": 56, "U": 57, "V": 58, "W": 59, "X": 60, "Y": 61,
        "Z": 62, "[": 63, "\\": 64, "]": 65, "^": 66, "_": 67, "`": 68,
        "a": 69, "b": 70, "c": 71, "d": 72, "e": 73, "f": 74, "g": 75, "h": 76,
        "i": 77, "j": 78, "k": 79, "l": 80, "m": 81, "n": 82, "o": 83, "p": 84,
        "q": 85, "r": 86, "s": 87, "t": 88, "u": 89, "v": 90, "w": 91, "x": 92,
        "y": 93, "z": 94, "{": 95, "|": 96, "}": 97, "~": 98, "¡": 99,
        "¢": 100, "£": 101, "¤": 102, "¥": 103, "¦": 104, "§": 105, "¨": 106,
        "©": 107, "ª": 108, "«": 109, "¬": 110, "®": 111, "¯": 112, "°": 113,
        "±": 114, "²": 115, "³": 116, "´": 117, "µ": 118, "¶": 119, "·": 120,
        "¸": 121, "¹": 122, "º": 123, "»": 124, "¼": 125, "½": 126, "¾": 127,
        "¿": 128, "À": 129, "Á": 130, "Â": 131, "Ã": 132, "Ä": 133, "Å": 134,
        "Æ": 135, "Ç": 136, "È": 137, "É": 138, "Ê": 139, "Ë": 140, "Ì": 141,
        "Í": 142, "Î": 143, "Ï": 144, "Ð": 145, "Ñ": 146, "Ò": 147, "Ó": 148,
        "Ô": 149, "Õ": 150, "Ö": 151, "×": 152, "Ø": 153, "Ù": 154, "Ú": 155,
        "Û": 156, "Ü": 157, "Ý": 158, "Þ": 159, "ß": 160, "à": 161, "á": 162,
        "â": 163, "ã": 164, "ä": 165, "å": 166, "æ": 167, "ç": 168, "è": 169,
        "é": 170, "ê": 171, "ë": 172, "ì": 173, "í": 174, "î": 175, "ï": 176,
        "ð": 177, "ñ": 178, "ò": 179, "ó": 180, "ô": 181, "õ": 182, "ö": 183,
        "÷": 184, "ø": 185, "ù": 186, "ú": 187, "û": 188, "ü": 189, "ý": 190,
        "þ": 191, "ÿ": 192, "Ā": 193, "ā": 194, "Ă": 195, "ă": 196, "Ą": 197,
        "ą": 198, "Ć": 199, "ć": 200, "Ĉ": 201, "ĉ": 202, "Ċ": 203, "ċ": 204,
        "Č": 205, "č": 206, "Ď": 207, "ď": 208, "Đ": 209, "đ": 210, "Ē": 211,
        "ē": 212, "Ĕ": 213, "ĕ": 214, "Ė": 215, "ė": 216, "Ę": 217, "ę": 218,
        "Ě": 219, "ě": 220, "Ĝ": 221, "ĝ": 222, "Ğ": 223, "ğ": 224, "Ġ": 225,
        "ġ": 226, "Ģ": 227, "ģ": 228, "Ĥ": 229, "ĥ": 230, "Ħ": 231, "ħ": 232,
        "Ĩ": 233, "ĩ": 234, "Ī": 235, "ī": 236, "Ĭ": 237, "ĭ": 238, "Į": 239,
        "į": 240, "İ": 241, "ı": 242, "Ĳ": 243, "ĳ": 244, "Ĵ": 245, "ĵ": 246,
        "Ķ": 247, "ķ": 248, "ĸ": 249, "Ĺ": 250, "ĺ": 251, "Ļ": 252, "ļ": 253,
        "Ľ": 254, "ľ": 255, "Ŀ": 256, "ŀ": 257, "Ł": 258, "ł": 259, "Ń": 260,
        "cc": 261, "CC": 262, "(=": 263, "ccc": 264, "](": 265, "@@": 266,
        "Cc": 267, "NC": 268, "ccccc": 269, "nc": 270, "CCC": 271, ")[": 272,
        "NH": 273, "+]": 274, "CO": 275, "cccc": 276, "Nc": 277, "Cl": 278,
        "OC": 279, "CCN": 280, ")(": 281, "COc": 282, "(-": 283, "([": 284,
        "CCCC": 285, "CN": 286, "-]": 287, ")(=": 288, "CCO": 289, "nH": 290,
        "nn": 291, "-])": 292, "+](": 293, "CCc": 294, ")=": 295, "sc": 296,
        "CS": 297, "ncc": 298, "Br": 299, "CNC": 300, "nnc": 301, "NCc": 302,
        "oc": 303, "12": 304, "+](=": 305, "CCCCC": 306, "COC": 307, "Cn": 308,
        "21": 309, "CCCN": 310, "cn": 311, "Oc": 312, "CCOC": 313,
        "CCOCC": 314, "+][": 315, "cnc": 316, "CCS": 317, "]([": 318,
        "CCOc": 319, "cccs": 320, "NCC": 321, "cccnc": 322, "OCC": 323,
        "CCCO": 324, "(/": 325, "@]": 326, "ccco": 327, "CSc": 328, "@@]": 329,
        "cnn": 330, "CCn": 331, "CCNC": 332, "32": 333, "ccccn": 334,
        "23": 335, "no": 336, "+])": 337, ")/": 338, "noc": 339, "csc": 340,
        "cs": 341, "ccncc": 342, "cccn": 343, "CCCc": 344, "Sc": 345,
        "ccnc": 346, "SCC": 347, "OCc": 348, "SC": 349, "ccn": 350,
        "ccsc": 351, "NNC": 352, "@](": 353, "OCO": 354, "NS": 355,
        "ncnc": 356, "NCCc": 357, "CNc": 358, "@@](": 359, "=[": 360,
        "OCCO": 361, "ncccc": 362, "NN": 363, "cncc": 364, "CCCCCC": 365,
        "NCCC": 366, "on": 367, "+]([": 368, "CCCCN": 369, "ncn": 370,
        "CCCNC": 371, "nccs": 372, "+]=": 373, "CSC": 374, "-])[": 375,
        "SCc": 376, "CCCn": 377, "sccc": 378, "cncn": 379, "CCSc": 380,
        "34": 381, "COCC": 382, "nnnn": 383, "nccc": 384, "(\\": 385,
        "ncccn": 386, "COCc": 387, "nccn": 388, ")([": 389, "CCSC": 390,
        "ccnn": 391, "ccoc": 392, "CNS": 393, "CCCOc": 394, "COCCN": 395,
        "43": 396, "@@](=": 397, "Fc": 398, "CCSCC": 399, "-])=": 400,
        "@](=": 401, "CSCC": 402, "CCCS": 403, "cnccn": 404, "nnn": 405,
        "/[": 406, "coc": 407, "nncn": 408, "cnnc": 409, "NCCN": 410,
        "NNc": 411, "nnnc": 412, "CCCCO": 413, "ncnn": 414, "+])[": 415,
        "CCl": 416, "Clc": 417, "OCCCO": 418, "CCNc": 419, "CSCc": 420,
        "cnnn": 421, "occc": 422, "CCCCNC": 423, "NCCNC": 424, "OCCC": 425,
        "CCNS": 426, "onc": 427, "CCCOC": 428, ")=[": 429, "nccnc": 430,
        "COCCn": 431, "OCCN": 432, "cnccc": 433, "FC": 434, "(\\[": 435,
        "CCCCCCC": 436, "NO": 437, "COCCNC": 438, "ns": 439, "cscc": 440,
        "13": 441, "cscn": 442, "nsc": 443, "NCCn": 444, "NCCOc": 445,
        "CCCCn": 446, "CCCCc": 447, "Nn": 448, "NCCCc": 449, "nonc": 450,
        "ccon": 451, "NCCCn": 452, "+])(": 453, "scnc": 454, "NCCS": 455,
        "NCCCN": 456, "ncsc": 457, "CNCc": 458, "CCCNc": 459, "NCCCC": 460,
        "Brc": 461, "scc": 462, "sccn": 463, "SCCC": 464, "COCCO": 465,
        "COCCOc": 466, "(=[": 467, "nncs": 468, "ocnc": 469, "CCOCc": 470,
        "31": 471, "nsnc": 472, "ncoc": 473, "OCCc": 474, "OS": 475,
        "SCCc": 476, "CCCCCN": 477, "OCCNC": 478, "CCCSc": 479, "CSCN": 480,
        "COCCCNC": 481, "COCCC": 482, "cncnc": 483, "CCCl": 484, "CCOCCN": 485,
        "-])/": 486, "co": 487, "CSCCS": 488, "nnsc": 489, "\\[": 490,
        "CCCCOc": 491, "CSCCO": 492, "NCCO": 493, "CBr": 494, "CCCCS": 495,
        ")-": 496, "COCCCN": 497, "NCCNc": 498, "+]\\": 499, "cccnn": 500,
        "](/": 501, "OCCn": 502, "CON": 503, "45": 504, "CCCSCC": 505,
        "csnn": 506, "OCCOc": 507, "@]([": 508, "+]/": 509, "SCCN": 510,
        ")(/": 511, "NCCCO": 512, "@@]([": 513, "OCCCC": 514, "CCCSC": 515,
        "ON": 516, "ncco": 517, "(/[": 518, "COCCc": 519, "OCOC": 520,
        "snc": 521, "ccncn": 522, "CSCCC": 523, "ccno": 524, "ncon": 525,
        "CCSCc": 526, "54": 527, "+])([": 528, "NCCSc": 529, "nnco": 530,
        "CCCCCNC": 531, "24": 532, "COCCOC": 533, "CSCCN": 534, "CCNCC": 535,
        "nncc": 536, "CCCOCC": 537, "NCCOC": 538, "NNS": 539, "CCCCOC": 540,
        "CONC": 541, "NOCc": 542, "NCCCOC": 543, "CCNCc": 544, "CNCC": 545,
        "SCCS": 546, "snnc": 547, "occ": 548, ")\\": 549, "COCCCC": 550,
        "CCCCl": 551, "OCCCc": 552, "NCCOCC": 553, "NCCCCn": 554,
        "COCCCn": 555, "SCCOc": 556, "ncncc": 557, "CNCCc": 558, "OCCCNC": 559,
        "CCOCCC": 560, "CCCNS": 561, "CCOCCNC": 562, "cncs": 563,
        "NCCCNC": 564, "CCCCSc": 565, "CCOCCO": 566, "42": 567, "OCCCN": 568,
        "CCCCCn": 569, "OCCNc": 570, "COCO": 571, "-])(": 572, "OCCOC": 573,
        "OCCSc": 574, "OCCS": 575, "NCN": 576, "OCN": 577, "NCCCSc": 578,
        "NCCNS": 579, "NOC": 580, "CCOCCCNC": 581, "cnco": 582, "COCCNc": 583,
        "CCCF": 584, "CCONC": 585, "NCCCCC": 586, "+]=[": 587, "-])=[": 588,
        "cnoc": 589, "OCCCn": 590, "]/": 591, "CNn": 592, "CCOCCCC": 593,
        "CCOCCn": 594, "COCCS": 595, "ClC": 596, "CCCCCc": 597, "SCCO": 598,
        "CNCCN": 599, "NCCCOc": 600, "NOCC": 601, "NCCCS": 602, "OCn": 603,
        "CCCCNc": 604, "CNN": 605, "SCCNC": 606, "ClCc": 607, "-]/": 608,
        "CCSCCC": 609, "OCCCOc": 610, "SCCn": 611, "CSCCNC": 612, ")/[": 613,
        "COCCCOc": 614, "CCOCCS": 615, "ccnnc": 616, "CCOCCOc": 617,
        "CSCCCNC": 618, "NCCCCN": 619, "sn": 620, "COCCOCC": 621, "-][": 622,
        "CCCCCO": 623, "SCCCS": 624, "ccs": 625, "-]=[": 626, "CCCCCS": 627,
        "cnns": 628, "COCCNS": 629, "COCCCNc": 630, "CSCCCCNC": 631,
        "CCBr": 632, "CSCCc": 633, "NCCCNc": 634, "NCCCOCC": 635, "SCCCC": 636,
        "oncc": 637, "CSCCCN": 638, "CSCCOC": 639, "CCCSCc": 640,
        "COCCSc": 641, "COn": 642, "(-[": 643, "NCCSCc": 644, "COCCCOC": 645,
        "COCCOCCNC": 646, "CCCCOCC": 647, "CCCCCCNC": 648, "CSCCn": 649,
        "CCCCNS": 650, "NCCSCC": 651, "][": 652, "CCCCCOc": 653, "Ic": 654,
        "NCCSC": 655, "OCCCCC": 656, "SCN": 657, "COP": 658, "CCOP": 659,
        "CSCCOc": 660, "cnsn": 661, "OCCCl": 662, "OCCCSc": 663, "nscc": 664,
        "COCOc": 665, "BrCc": 666, "NCCCCl": 667, "OP": 668, "SCn": 669,
        "SCCCc": 670, "](/[": 671, "COS": 672, "CCCCCCN": 673,
        "CCOCCCN": 674, "+](-": 675, "+](/": 676, "CCCOCc": 677, "CI": 678,
        "NOc": 679, "NCCCCCC": 680, "OCCNS": 681, "SN": 682, "BrC": 683,
        "cnsc": 684, "OCCF": 685, "35": 686, "OCCSC": 687, "SCCOC": 688,
        "SCCCO": 689, "SCCCOc": 690, "COCCCCC": 691, "ClCC": 692, "-])([": 693,
        "CCCCCOC": 694, "CCOCCNc": 695, "OCCCNc": 696, "NCCCCOc": 697,
        "NCCOCc": 698, "OCCOCC": 699, "OCCCS": 700, "OCCSCc": 701,
        "SCCCN": 702, "ClCCc": 703, "CCCCCCO": 704, "CCCCCCCCCCC": 705,
        "CNCCC": 706, "CCCNCC": 707, "CCOCCSc": 708, "NNN": 709, "NCCCSC": 710,
        "COCCNCc": 711, "41": 712, "56": 713, "NSC": 714, "NCCCCc": 715,
        "ONC": 716, "conc": 717, "CCCBr": 718, "+]/[": 719, "CCCCCCn": 720,
        "CCOCCOC": 721, "CCOCCOCC": 722, "CCOCCCn": 723, "CCOCCCNc": 724,
        "BrCC": 725, "CCCCCl": 726, "CCCCCSc": 727, "CCSS": 728,
        "CCSCCOC": 729, "OCCCCN": 730, "NCCCNS": 731, "NCCCOCc": 732,
        "OH": 733, "SCCSc": 734, "NCn": 735, "CCCCCCS": 736, "CCCCOCc": 737,
        "CCCCSC": 738, "CSCCSC": 739, "CCSCCN": 740, "OCCBr": 741,
        "OCCOCCOCCO": 742, "NCCCCSC": 743, "COCCOCCN": 744, "NCCOCCO": 745,
        "(#": 746, "14": 747, "IC": 748, "On": 749, "OCCSCC": 750,
        "OCCCCn": 751, "OCCOCCS": 752, "SS": 753, "SSC": 754, "SCCCn": 755,
        "NCNS": 756, "COCCCS": 757, "COCCCCCNC": 758, "CCCCSCC": 759,
        "CCCCCCSc": 760, "CCON": 761, "nnccc": 762, "-])\\": 763,
        "+](\\[": 764, "CSCCCC": 765, "CSCCCNc": 766, "123": 767,
        "CCOCCCc": 768, "CCOCCOCCOCC": 769, "-[": 770, "132": 771, "53": 772,
        "CH": 773, "NP": 774, "OO": 775, "OCOc": 776, "PH": 777, "SCCNS": 778,
        "sncc": 779, "CCCNCc": 780, "+])=": 781, "COCCCc": 782, "CONS": 783,
        "COCCCCNC": 784, "COCCCCS": 785, "CCCCCCCC": 786, "CCCCCCc": 787,
        "CCOCCOCc": 788, "CSCCCn": 789, "BrCCC": 790, "BrCCc": 791,
        "CCCCCBr": 792, "CCSCCOc": 793, "CCSCCCO": 794, "CCSCCn": 795,
        "OCCl": 796, "CCCOCCO": 797, "+])/": 798, "SCCNc": 799,
        "OCCOCCOc": 800, "CCCCCCNc": 801, "NCCCF": 802, "NCCCl": 803,
        "CCCCOCCN": 804, "NCCCCCc": 805, "OCCCCSc": 806, "COCCSCCC": 807,
        "#[": 808, "312": 809, "CF": 810, "FO": 811, "FCCC": 812, "NNCc": 813,
        "NSc": 814, "ONc": 815, "OCCCCNC": 816, "OCCCNS": 817, "OCCCCCS": 818,
        "SH": 819, "SSc": 820, "SCCCCO": 821, "SCCCCCO": 822, "SCCCCCS": 823,
        "SSN": 824, "ss": 825, "ssc": 826, "ssnc": 827, "NCSc": 828,
        "ncno": 829, "CCCOCCNC": 830, "+]#": 831, "+]\\[": 832, "COCCCCN": 833,
        "COCCSCC": 834, "COCCCCOc": 835, "COCCSCc": 836, "ClCCCSc": 837,
        "CCCCCCOc": 838, "CCCCSCc": 839, "CCCCOCCNC": 840, "CNNC": 841,
        "CNCCO": 842, "-]/[": 843, "CCOS": 844, "CCOCO": 845, "CCOCCc": 846,
        "CCOCOCC": 847, "+](/[": 848, "CSCCCc": 849, "CSCCNS": 850,
        "CCCCCNc": 851, "CCCCCOCC": 852, "CCCCCNS": 853, "CCSCCc": 854,
        "CCSCCSc": 855, "NCCBr": 856, "CCCOS": 857, "CCCOCCC": 858,
        "nocc": 859, "@@](/": 860, "OCCOCc": 861, "OCCOCCN": 862,
        "OCCOCCO": 863, "CCCCCCl": 864, "NCCCSCC": 865, "CCCCOCCCNC": 866,
        "OCCCOC": 867, "OCCCBr": 868, "OCCCSC": 869, "NOCCc": 870,
        "SCCCBr": 871, "SCCCSCC": 872, "COCCOCc": 873, "NCCOCCc": 874,
        "COCCCNS": 875, "COCCOCCSc": 876, "SCCCCCCSc": 877,
        "OCCOCCOCCOCCO": 878
    }

    reverse_vocab = {}
    for val, key in vocab.items():
        reverse_vocab[key] = val

    return vocab, reverse_vocab


def tokens_to_smiles(tokens):
    """
    """

    vocab, reversed = get_vocab()
    smiles = []
    for token in tokens:
        smiles.append(reversed[token])
    return smiles
