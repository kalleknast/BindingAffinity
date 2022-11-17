# BindingAffinity
An implementation of the paper "Modelling Drug-Target Binding Affinity using a BERT based Graph Neural network"

## TODO
 - [ ] fix `total=raw.shape[0]` (`tqdm(raw.iterrows(), total=raw.shape[0])`) in `BindingAffinityDataset()` line 99. The total number of embedding files is only 447.
 - [ ] fix data split issue. Right now the data is re-split each time `BindingAffinityDataset()` is instantiated which leads to overlap between the trainin and testing data.
 - [ ] see if it is possible to remove the master files in `BindingAffinityDataset()`.

