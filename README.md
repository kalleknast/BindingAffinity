# BindingAffinity
Implementation and comparison of a few models predicting the binding affinity between a small molecule (i.e. a drug) and a protein. The models are fitted on the [Kiba dataset](https://pubs.acs.org/doi/10.1021/ci400709d).

## Introduction

### DeepDTA
The DeepDTA model is presented in [DeepDTA: Deep Drug-Target Binding Affinity Prediction](https://arxiv.org/abs/1801.10193). The original code is available [here](https://github.com/hkmztrk/DeepDTA).

### BERT-GCN
The paper [Modelling Drug-Target Binding Affinity using a BERT based Graph Neural network](https://openreview.net/pdf?id=Zqf6RGp5lqf) by Lennox, Robertson and Devereux presents a graph convolutional neural network (GCN) trained to predict the binding affinity of drugs to proteins. Their model takes as input BERT-embedded protein sequences and drug molecules. This combination of BERT embeddings and a graph network is relatively novel, and the model achieves (at publication) state-of-the-art results. However, the paper leaves many technical details unspecified, and no code is provided. Thus, the goal is the implement the GCN and replicate the results from the paper.

## Install
```termial
pip install -r requirements.txt
```

## Issues and solutions

### Data
A GCN takes nodes and edges as inputs. The paper describes embedding both proteins and drugs using pre-trained BERT models where each token is embedded as a 768-long vector. This results in two issues that are not mentioned in the paper.
### Proteins
The primary protein sequence is tokenized into amino acids and there are trivial edges between neighbouring tokens. However, a protein's 3D structure puts some amino acids very close to each other and edges between those should also be included. The authors do not mention anything about this. Here, only edges to neighbours in the primary sequence are included.
### Drugs
The drugs are tokenized with a Byte-Pair Encoder (BPE) from Simplified molecular-input line-entry system [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system). A BPE combines characters to produce a fixed-size vocabulary where frequently occurring subsequences of characters are combined into tokens. The particular BPE (probably) used in the paper separates bond tokens and atom tokens so that multi-character tokens are made of either only atoms (nodes) or only bonds. An embedded SMILES will be made up of both node (atoms) and bond vectors. Since the bonds often correspond to small groups of atoms, the edges computed directly from a SMILES string do not match (the latter edges are between atoms). The paper does not specify how this was resolved. Here, only edges between nodes were included, and embedding vectors corresponding to bonds were removed (i.e. only node vectors were included as inputs to the GCN).

### Network architecture
The network architecture is described in Fig. 1 of the paper.
 1. **Issue**: In step 1, there is an average pooling layer after embedding (both protein and drug). This layer collapses the tensors over the nodes and is generally used as a readout layer right before the classification/regression head. The purpose of the average pooling layer before the GCN layers is unclear. **Solution**: This average pooling layer was omitted.
 2. **Issue**:In step 4, there is a concatenation layer directly after the GCN layers. What is being concatenated is unclear. **Solution**: This concatenation layer was omitted.
 3. **Issue**: In step 3, there is no readout layer that collapses over nodes. Thus, the input to the final dense layers will have a variable number of nodes. This does not work. **Solution**: After the GCN layers, an average pooling layer was added.
 
## Use
In the working directory, make the directory `data,` and then the subdirectories `raw` and `processed` in `data`.
```terminal
mkdir -p data/raw
mkdir -p data/processed
```

Train and evaluate the GCN model (this will download and process the data the first time it runs):
```terminal
python gcn_model.py
```

Train and evaluate the MLP model:
```terminal
python mlp_model.py
```

 

## TODO
 - [x] fix `total=raw.shape[0]` (`tqdm(raw.iterrows(), total=raw.shape[0])`) in `BindingAffinityDataset()` line 99. The total number of embedding files is only 447.
 - [x] fix data split issue. Right now the data is re-split each time `BindingAffinityDataset()` is instantiated, which leads to an overlap between the training and testing data.
   - Don't use `split = raw_data.get_split()` (line 75). Just preprocess all data and split in torch_geometric.
 - [x] see if it is possible to remove the master files in `BindingAffinityDataset()`.
 - [x] Sometimes `torch_geometric.utils.from_smiles` returns `edge_index` with an edge to one more node than the number of nodes in the embedding. This is probably because of a discrepancy between the BPE tokenizer (`transformers.AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k")`) and the atoms in the SMILES string.
 - [ ] See if the actual `edge_index` for the proteins can be downloaded from the [UniProt](https://www.uniprot.org/) protein database.
 - [ ] Add residual connections to the GCN layers.

