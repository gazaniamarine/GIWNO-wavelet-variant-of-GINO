import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from neuralop.models.gino import GINO
from neuralop.utils.losses import LpLoss
import matplotlib.pyplot as plt
import os
import sys

# Import GINO config
sys.path.append('/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO')
from config.gino_config import config

class PoissonDonutDataset(Dataset):
    def __init__(self, file_path, n_samples=None):
        data = torch.load(file_path)
        self.coords = data['coords']
        self.inputs = data['inputs']
        self.outputs = data['outputs']
        self.latent_grid = data['latent_grid']
        if n_samples is not None:
            self.inputs = self.inputs[:n_samples]
            self.outputs = self.outputs[:n_samples]
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.outputs[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training GINO (FNO Baseline) on {device}")

    data_cfg = config['data']
    full_dataset = PoissonDonutDataset(data_cfg['path'], data_cfg.get('n_samples'))
    train_size = int(data_cfg['train_test_split'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    model_cfg = config['model']
    model = GINO(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        gno_coord_dim=model_cfg['gno_coord_dim'],
        fno_n_modes=model_cfg['fno']['n_modes'],
        fno_hidden_channels=model_cfg['fno']['hidden_channels'],
        fno_n_layers=model_cfg['fno']['n_layers'],
        fno_in_channels=model_cfg['fno']['in_channels'],
        in_gno_radius=model_cfg['gno']['in_radius'],
        out_gno_radius=model_cfg['gno']['out_radius'],
        gno_use_open3d=model_cfg['gno']['use_open3d']
    ).to(device)

    train_cfg = config['train']
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['scheduler_t_max'])
    criterion = LpLoss(d=2, p=2, reduction=True)

    coords = full_dataset.coords.unsqueeze(0).to(device)
    latent_grid = full_dataset.latent_grid.unsqueeze(0).to(device)

    epochs = train_cfg['epochs']
    history = {'train_loss': [], 'test_loss': []}
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(input_geom=coords, latent_queries=latent_grid, output_queries=coords, x=x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(coords, latent_grid, coords, x)
                test_loss += criterion(out, y).item()
        avg_train, avg_test = train_loss/len(train_loader), test_loss/len(test_loader)
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | RelL2 Train: {avg_train:.6f} | RelL2 Test: {avg_test:.6f}")

    model_path = os.path.join(save_dir, config['output']['model_name'])
    torch.save(model.state_dict(), model_path)
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.yscale('log')
    plt.title("GINO-FNO Training Loss")
    plt.savefig(os.path.join(save_dir, 'loss_gino_fno.png'))
    print(f"Completed. Model saved to {model_path}")

if __name__ == "__main__":
    train()
