# BindingAffinity
An implementation of the paper [Modelling Drug-Target Binding Affinity using a BERT based Graph Neural network](https://openreview.net/pdf?id=Zqf6RGp5lqf).

## TODO
 - [x] fix `total=raw.shape[0]` (`tqdm(raw.iterrows(), total=raw.shape[0])`) in `BindingAffinityDataset()` line 99. The total number of embedding files is only 447.
 - [x] fix data split issue. Right now the data is re-split each time `BindingAffinityDataset()` is instantiated which leads to overlap between the trainin and testing data.
   - Don't use `split = raw_data.get_split()` (line 75). Just preprocess all data and split in torch_geometric.
 - [x] see if it is possible to remove the master files in `BindingAffinityDataset()`.
 - [ ] Sometimes `torch_geometric.utils.from_smiles` returns `edge_index` with an edge to one more node than the number of nodes in the embedding. This is probably because of a discrepancy between the BPE tokenizer (`transformers.AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k")`) and the atoms in the SMILES string.
 - [ ] See if actual `edge_index` for the proteins can be downloaded from the [UniProt](https://www.uniprot.org/) protein database.

