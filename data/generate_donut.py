import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def generate_donut_data(n_samples=200, resolution=128, t_final=0.1, dt=0.0005, nu=0.01):
    """
    Generates 2D Burgers' data masked to a donut geometry.
    Eq: u_t + u*u_x + v*u_y = nu * div(grad(u))
        v_t + u*v_x + v*v_y = nu * div(grad(v))
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Grid
    x = torch.linspace(-1, 1, resolution, device=device)
    y = torch.linspace(-1, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    
    # Donut Mask: 0.3 < radius < 0.9
    mask = (R > 0.3) & (R < 0.9)
    irreg_coords = torch.stack([X[mask], Y[mask]], dim=-1) # (N_pts, 2)
    
    # 2. Solver Parameters
    dx = x[1] - x[0]
    n_steps = int(t_final / dt)
    
    all_inputs = []  # ICs at t=0
    all_outputs = [] # Solutions at t=t_final
    
    print(f"Generating {n_samples} samples on {device}...")
    
    for i in tqdm(range(n_samples)):
        # Generate Random Initial Conditions (Smooth Gaussian Fields)
        # Using a simple spectral approach for smooth ICs
        u = torch.randn(resolution, resolution, device=device)
        v = torch.randn(resolution, resolution, device=device)
        
        # Apply low-pass filter to smooth ICs
        u_f = torch.fft.fftn(u)
        v_f = torch.fft.fftn(v)
        freqs = torch.fft.fftfreq(resolution, device=device)
        FX, FY = torch.meshgrid(freqs, freqs, indexing='ij')
        filter_mask = torch.exp(-20 * (FX**2 + FY**2)) # Smoother ICs
        u = torch.real(torch.fft.ifftn(u_f * filter_mask))
        v = torch.real(torch.fft.ifftn(v_f * filter_mask))
        
        # Normalize ICs to have reasonable magnitude
        u = u / u.std() * 0.5
        v = v / v.std() * 0.5
        
        u_start = u.clone()
        v_start = v.clone()
        
        # 3. Time Stepping (Simple Explicit Finite Difference)
        # Using periodic rolling for boundaries (which are far from the donut)
        for _ in range(n_steps):
            # Gradients using central differences
            u_x = (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2*dx)
            u_y = (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2*dx)
            v_x = (torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2*dx)
            v_y = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2*dx)
            
            # Laplacian
            lap_u = (torch.roll(u, -1, 0) + torch.roll(u, 1, 0) + 
                     torch.roll(u, -1, 1) + torch.roll(u, 1, 1) - 4*u) / (dx**2)
            lap_v = (torch.roll(v, -1, 0) + torch.roll(v, 1, 0) + 
                     torch.roll(v, -1, 1) + torch.roll(v, 1, 1) - 4*v) / (dx**2)
            
            u = u - dt * (u*u_x + v*u_y) + nu * dt * lap_u
            v = v - dt * (u*v_x + v*v_y) + nu * dt * lap_v
            
        # Sample points in the donut
        input_sample = torch.stack([u_start[mask], v_start[mask]], dim=-1)
        output_sample = torch.stack([u[mask], v[mask]], dim=-1)
        
        all_inputs.append(input_sample.cpu())
        all_outputs.append(output_sample.cpu())

    # Save data
    data = {
        'coords': irreg_coords.cpu(), # Fixed for all samples, shape (N_pts, 2)
        'inputs': torch.stack(all_inputs), # (B, N_pts, 2)
        'outputs': torch.stack(all_outputs), # (B, N_pts, 2)
        'latent_grid': torch.stack([X, Y], dim=-1).cpu(), # (Resolution, Resolution, 2)
        'mask': mask.cpu()
    }
    
    save_path = '/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/data/burgers_donut.pt'
    torch.save(data, save_path)
    print(f"\nSaved dataset to {save_path}")
    print(f"Number of points in donut: {len(irreg_coords)}")

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.scatter(irreg_coords[:, 0].cpu(), irreg_coords[:, 1].cpu(), c=all_inputs[0][:, 0], s=2)
    plt.title("Initial Condition (u-component)")
    plt.colorbar(im1)
    
    plt.subplot(1, 2, 2)
    im2 = plt.scatter(irreg_coords[:, 0].cpu(), irreg_coords[:, 1].cpu(), c=all_outputs[0][:, 0], s=2)
    plt.title("Solution at t=0.1 (u-component)")
    plt.colorbar(im2)
    
    plt.savefig('/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/data/donut_sample.png')
    print("Saved sample visualization to data/donut_sample.png")

if __name__ == "__main__":
    generate_donut_data()
