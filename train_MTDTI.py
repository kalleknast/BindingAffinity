import torch
from torch.utils.data import DataLoader
from data import EmbeddingDataset
from utils import train, evaluate
from plot import plot_predictions, plot_losses
from models import MTDTI
import json

model_name = 'MTDTI-zinc250k'
dataset_name = 'KIBA'
root = 'data'
epochs = 200
# partition_kind = 'pair'
partition_kind = 'drug'
batch_size = 256
# drug_encoder = 'DeepChem/ChemBERTa-77M-MTR'
# drug_encoder = 'DeepChem/ChemBERTa-77M-MLM'
drug_encoder = "seyonec/ChemBERTa_zinc250k_v2_40k"
# drug_encoder = 'seyonec/PubChem10M_SMILES_BPE_50k'

model_name = f'{model_name}_{partition_kind}split'
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

dataset = EmbeddingDataset(root, dataset_name, partition_kind=partition_kind,
                           drug_tokenizer=drug_encoder,
                           prot_tokenizer='DeepDTA')
data_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True)

model = MTDTI(dataset, drug_encoder=drug_encoder).to(device)
# Some weights of RobertaModel were not initialized from the model checkpoint
# at DeepChem/ChemBERTa-77M-MTR and are newly initialized:
# ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
#
# Freeze all but the last drug_encoder/BERT pooler layer
model.drug_encoder.embeddings.requires_grad_(False)
model.drug_encoder.encoder.requires_grad_(False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()


history = {'train_loss': [], 'valid_loss': []}
for epoch in range(1, epochs+1):

    train_loss = train(data_loader, model, device,
                       loss_fn, optimizer, verbose=True)
    pred_valid, y_valid, valid_loss = evaluate(data_loader, model, device,
                                               loss_fn, partition='valid')
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)

    print(f'Epoch: {epoch:02d} -- train MSE: {train_loss:.4f}'
          f', valid MSE: {valid_loss:.4f}')

pred_train, y_train, train_loss = evaluate(data_loader, model, device,
                                           loss_fn, partition='train')

plot_predictions(y_train, pred_train, y_valid, pred_valid, model_name, epoch)
plot_losses(history, model_name)
# Save the losses
with open(f'results/history_{model_name}_epochs{epoch}.json', 'w') as f:
    json.dump(history, f)
