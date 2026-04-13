import os
import sys
# Ensure the root directory is prioritized in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from neuralop.models.gino import GINO
from neuralop.utils.losses import LpLoss
import matplotlib.pyplot as plt
import time

# Import config
from config.poisson_gino_donut_config import config

# 1. Dataset for Poisson Donut
class PoissonDonutDataset(Dataset):
    def __init__(self, file_path, n_samples=None):
        data = torch.load(file_path)
        self.coords = data['coords'] # (N_pts, 2)
        self.inputs = data['inputs'] # (B, N_pts, 4): [f, x, y, sdf]
        self.outputs = data['outputs'] # (B, N_pts, 1)
        self.latent_grid = data['latent_grid'] # (Res, Res, 2)
        self.latent_sdf = data.get('latent_sdf') # (Res, Res, 1) or None
        
        if n_samples is not None:
            self.inputs = self.inputs[:n_samples]
            self.outputs = self.outputs[:n_samples]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# 2. Training Setup
def train():
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        latent_feature_channels=1 if full_dataset.latent_sdf is not None else None,
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
    latent_sdf = full_dataset.latent_sdf.unsqueeze(0).to(device) if full_dataset.latent_sdf is not None else None # (1, Res, Res, 1)

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
                x=x,
                latent_features=latent_sdf
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
                out = model(coords, latent_grid, coords, x, latent_features=latent_sdf)
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
    plt.savefig(os.path.join(save_dir, 'gino_training_loss.png'))
    print(f"Training Complete. Model saved to {model_path}")

    # Visualize results
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x, y = x[:1].to(device), y[:1].to(device)
        pred = model(coords, latent_grid, coords, x, latent_features=latent_sdf)
        
        # Pointwise absolute error to see where model fails
        error = torch.abs(y[0, :, 0] - pred[0, :, 0]).cpu()
        y_cpu = y[0, :, 0].cpu()
        pred_cpu = pred[0, :, 0].cpu()
        coords_cpu = coords[0].cpu()

        plt.figure(figsize=(20, 5))
        
        # Subplot 1: True u
        plt.subplot(1, 3, 1)
        im1 = plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], c=y_cpu, s=2, cmap='viridis')
        plt.title("True Solution u(x)")
        plt.colorbar(im1)
        plt.axis('equal')
        
        # Subplot 2: Predicted u
        plt.subplot(1, 3, 2)
        im2 = plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], c=pred_cpu, s=2, cmap='viridis')
        plt.title("Predicted Solution u(x)")
        plt.colorbar(im2)
        plt.axis('equal')
        
        # Subplot 3: Error
        plt.subplot(1, 3, 3)
        im3 = plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], c=error, s=2, cmap='magma')
        plt.title("Pointwise Absolute Error")
        plt.colorbar(im3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gino_prediction_sample.png'))
        print(f"Visualization saved to {os.path.join(save_dir, 'prediction_sample.png')}")

    # Print final error
    final_test_loss = history['test_loss'][-1]
    print(f"\n" + "="*50)
    print(f"FINAL TRAINING SUMMARY")
    print(f"Final Relative L2 Test Error: {final_test_loss:.6f}")
    print("="*50)

if __name__ == "__main__":
    train()
