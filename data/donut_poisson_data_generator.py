import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def generate_poisson_donut(n_samples=500, resolution=128):
    """
    Generates 2D Poisson data on a donut geometry.
    Eq: -Laplacian(u) = f, with u=0 on the boundary.
    Uses the inverse method: define u, compute f.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Grid
    x = torch.linspace(-1, 1, resolution, device=device)
    y = torch.linspace(-1, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    dx = x[1] - x[0]
    
    # Donut Mask: 0.3 < radius < 0.9
    mask = (R > 0.3) & (R < 0.9)
    # Smooth mask for zero boundary conditions (sin^2 transition)
    boundary_mask = torch.where(mask, 
                                torch.sin(np.pi * (R - 0.3) / 0.6)**2, 
                                torch.zeros_like(R))
    
    # Subsample points to ~2500 for faster training
    n_pts_target = 2500
    n_pts_total = mask.sum().item()
    indices = np.random.choice(n_pts_total, n_pts_target, replace=False)
    
    irreg_coords = torch.stack([X[mask], Y[mask]], dim=-1)[indices] # (N_pts, 2)
    
    all_f = [] # Inputs (Source Term)
    all_u = [] # Outputs (Potential)
    
    print(f"Generating {n_samples} Poisson samples on {device}...")
    
    for i in tqdm(range(n_samples)):
        # ... (rest of the sample generation logic)
        u_raw = torch.randn(resolution, resolution, device=device)
        u_f = torch.fft.fftn(u_raw)
        freqs = torch.fft.fftfreq(resolution, device=device)
        FX, FY = torch.meshgrid(freqs, freqs, indexing='ij')
        filter_mask = torch.exp(-10 * (FX**2 + FY**2))
        u_global = torch.real(torch.fft.ifftn(u_f * filter_mask))
        u_global = u_global / u_global.std() * 0.5
        
        u_local = torch.zeros_like(u_global)
        n_spikes = np.random.randint(1, 4)
        for _ in range(n_spikes):
            angle = np.random.rand() * 2 * np.pi
            radius = 0.4 + np.random.rand() * 0.4
            cx, cy = radius * np.cos(angle), radius * np.sin(angle)
            sigma = 0.02 + np.random.rand() * 0.03
            amp = (np.random.rand() - 0.5) * 2.0
            pulse = amp * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            u_local += pulse
            
        u = u_global + u_local
        u = u * boundary_mask
        
        lap_u = (torch.roll(u, -1, 0) + torch.roll(u, 1, 0) + 
                 torch.roll(u, -1, 1) + torch.roll(u, 1, 1) - 4*u) / (dx**2)
        f = -lap_u
        
        # Sample only the selected indices
        all_f.append(f[mask][indices].cpu())
        all_u.append(u[mask][indices].cpu())

    # Save data
    data = {
        'coords': irreg_coords.cpu(), # Fixed for all samples
        'inputs': torch.stack(all_f).unsqueeze(-1).float(), # (B, N_pts, 1)
        'outputs': torch.stack(all_u).unsqueeze(-1).float(), # (B, N_pts, 1)
        'latent_grid': torch.stack([X, Y], dim=-1).cpu().float() # (Res, Res, 2)
    }
    
    os.makedirs('data', exist_ok=True)
    save_path = '/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/data/poisson_donut.pt'
    torch.save(data, save_path)
    print(f"\nSaved Poisson dataset to {save_path}")
    print(f"Number of points: {len(irreg_coords)}")

    # Visualization of a spike sample
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(irreg_coords[:, 0].cpu(), irreg_coords[:, 1].cpu(), c=all_f[-1], s=1, cmap='inferno')
    plt.title("Source Term (f) with Local Spikes")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.scatter(irreg_coords[:, 0].cpu(), irreg_coords[:, 1].cpu(), c=all_u[-1], s=1, cmap='viridis')
    plt.title("Solution (u) - Zero BC on boundaries")
    plt.colorbar()
    
    plt.savefig('/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/data/poisson_sample.png')
    print("Saved sample visualization to data/poisson_sample.png")

if __name__ == "__main__":
    generate_poisson_donut(n_samples=1000) # 1000 samples for better training
