import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.data import DataLoader
from data import MTDTIDataset, EmbeddingDataset
from utils import train, evaluate
from plot import plot_predictions, plot_losses
from models import MTDTI
import json

model_name = 'MTDTI'
dataset_name = 'KIBA'
root = 'data'
epochs = 100
partition_kind = 'pair'
# partition_kind = 'drug'
batch_size = 64

model_name = f'{model_name}_{partition_kind}split'
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

# dataset = MTDTIDataset(root, dataset_name=dataset_name,
#                        partition_kind=partition_kind)
# data_loader = GraphDataLoader(dataset,
#                               batch_size=batch_size,
#                               shuffle=True)
# Setting num_workers seems to result in the following error:
# RuntimeError: torch.cat(): input types can't be cast to the desired output
# type Long
dataset = EmbeddingDataset(root, dataset_name, partition_kind=partition_kind,
                           drug_tokenizer='BERT-drug', prot_tokenizer='DeepDTA')
data_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True)

model = MTDTI(dataset).to(device)
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
