import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from neuralop.models.gino import GINO
from neuralop.utils.losses import LpLoss
import matplotlib.pyplot as plt
import time
import os
import sys

# Import config
sys.path.append('/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO')
from config.poisson_gino_donut_config import config

# 1. Dataset for Poisson Donut
class PoissonDonutDataset(Dataset):
    def __init__(self, file_path, n_samples=None):
        data = torch.load(file_path)
        self.coords = data['coords'] # (N_pts, 2)
        self.inputs = data['inputs'] # (B, N_pts, 1)
        self.outputs = data['outputs'] # (B, N_pts, 1)
        self.latent_grid = data['latent_grid'] # (Res, Res, 2)
        
        if n_samples is not None:
            self.inputs = self.inputs[:n_samples]
            self.outputs = self.outputs[:n_samples]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# 2. Training Setup
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load Data from Config
    data_cfg = config['data']
    full_dataset = PoissonDonutDataset(data_cfg['path'], data_cfg.get('n_samples'))
    train_size = int(data_cfg['train_test_split'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg['batch_size'], shuffle=False)

    # Initialize GINO Model from Config
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
        gno_use_open3d=model_cfg['gno']['use_open3d'],
    ).to(device)

    # Optimizer & Scheduler from Config
    train_cfg = config['train']
    optimizer = optim.Adam(model.parameters(), 
                           lr=train_cfg['learning_rate'], 
                           weight_decay=train_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['scheduler_t_max'])
    
    # Loss Function (Relative L2 per request)
    if train_cfg['loss_type'] == 'rel_l2':
        criterion = LpLoss(d=2, p=2, reduction=True)
    else:
        criterion = nn.MSELoss()

    # Pre-fetch geometry
    coords = full_dataset.coords.unsqueeze(0).to(device) # (1, N_pts, 2)
    latent_grid = full_dataset.latent_grid.unsqueeze(0).to(device) # (1, Res, Res, 2)

    # 3. Training Loop
    epochs = train_cfg['epochs']
    history = {'train_loss': [], 'test_loss': []}
    
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # GINO forward
            out = model(
                input_geom=coords,
                latent_queries=latent_grid,
                output_queries=coords,
                x=x
            )
            
            # Using Rel L2 Loss
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(coords, latent_grid, coords, x)
                test_loss += criterion(out, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | RelL2 Train: {avg_train:.6f} | RelL2 Test: {avg_test:.6f}")

    # 4. Save results
    model_path = os.path.join(save_dir, config['output']['model_name'])
    torch.save(model.state_dict(), model_path)
    
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title("GINO-FNO Training (Poisson Donut)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    print(f"Training Complete. Model saved to {model_path}")

    # Visualize results
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x, y = x[:1].to(device), y[:1].to(device)
        pred = model(coords, latent_grid, coords, x)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(coords[0, :, 0].cpu(), coords[0, :, 1].cpu(), c=x[0, :, 0].cpu(), s=1)
        plt.title("Input f(x)")
        plt.subplot(1, 3, 2)
        plt.scatter(coords[0, :, 0].cpu(), coords[0, :, 1].cpu(), c=y[0, :, 0].cpu(), s=1)
        plt.title("Expected u(x)")
        plt.subplot(1, 3, 3)
        plt.scatter(coords[0, :, 0].cpu(), coords[0, :, 1].cpu(), c=pred[0, :, 0].cpu(), s=1)
        plt.title("Predicted (FNO Baseline)")
        plt.savefig(os.path.join(save_dir, 'prediction_sample.png'))

if __name__ == "__main__":
    train()
