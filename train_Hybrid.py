import torch
from torch_geometric.loader import DataLoader
from data import HybridDataset
from utils import train, evaluate
from plot import plot_predictions, plot_losses
from models import Hybrid
import json

model_name = 'Hybrid'
dataset_name = 'KIBA'
root = 'data'
partition_kind = 'drug'
epochs = 20
batch_size = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

dataset = HybridDataset(root,
                        dataset_name=dataset_name,
                        partition_kind=partition_kind)
data_loader = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=12)

model = Hybrid(dataset, 96).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
