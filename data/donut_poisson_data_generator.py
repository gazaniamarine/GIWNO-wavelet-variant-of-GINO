"""Generates Poisson equation data on an annular domain using the Method of Manufactured Solutions."""

import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi
from tqdm import tqdm

def evaluate_basis(r, theta, k, m, r_in, r_out, is_cos):
    """Evaluate a single polar basis function: sin(k*pi*rho) * {cos/sin}(m*theta)."""
    L = r_out - r_in
    rho = (r - r_in) / L
    R_k = np.sin(k * pi * rho)
    Phi_m = np.cos(m * theta) if is_cos else np.sin(m * theta)
    return R_k * Phi_m

def laplacian_basis(r, theta, k, m, r_in, r_out, is_cos):
    """Compute the analytical Laplacian of the polar basis function."""
    L = r_out - r_in
    rho = (r - r_in) / L
    R_k = np.sin(k * pi * rho)
    dR_k = (k * pi / L) * np.cos(k * pi * rho)
    d2R_k = -(k * pi / L) ** 2 * np.sin(k * pi * rho)
    Phi_m = np.cos(m * theta) if is_cos else np.sin(m * theta)
    return (d2R_k + dR_k / r - (m ** 2 / r ** 2) * R_k) * Phi_m

def sample_annulus_uniform(n_points, r_in, r_out, seed=None):
    """Sample points uniformly by area inside the annulus."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0, 1, n_points)
    r = np.sqrt(r_in ** 2 + U * (r_out ** 2 - r_in ** 2))
    theta = rng.uniform(0, 2 * pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, r, theta

import torch

def generate_poisson_donut(
    n_train=200, n_test=30, n_points=3000, r_in=0.4, r_out=1.0, 
    K_max=4, M_max=4, out_dir=None, pt_path=None, seed=42
):
    """Generate exact Poisson equation data on an annular domain and save as .txt and .pt files."""
    base = os.path.dirname(os.path.abspath(__file__))
    if out_dir is None:
        out_dir = os.path.join(base, "..", "support_files", "donut_poisson_mms")
    if pt_path is None:
        pt_path = os.path.join(base, "poisson_donut.pt")

    train_dir, test_dir = os.path.join(out_dir, "train"), os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    x_irreg, y_irreg, r, theta = sample_annulus_uniform(n_points, r_in, r_out, seed=seed)
    coords = torch.from_numpy(np.column_stack([x_irreg, y_irreg])).float()

    # Latent Grid for GINO (FNO)
    res = 64
    gx = np.linspace(-1, 1, res)
    gy = np.linspace(-1, 1, res)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    latent_grid = torch.from_numpy(np.stack([GX, GY], axis=-1)).float()

    np.savetxt(os.path.join(out_dir, "coordinates.txt"), coords.numpy(), 
               header="x y", fmt="%.10f", comments="")

    terms = []
    for k in range(1, K_max + 1):
        terms.append((k, 0, True))
        for m in range(1, M_max + 1):
            terms.extend([(k, m, True), (k, m, False)])

    n_terms = len(terms)
    basis_vals = np.zeros((n_terms, n_points))
    basis_laps = np.zeros((n_terms, n_points))
    for idx, (k, m, is_cos) in enumerate(terms):
        basis_vals[idx] = evaluate_basis(r, theta, k, m, r_in, r_out, is_cos)
        basis_laps[idx] = laplacian_basis(r, theta, k, m, r_in, r_out, is_cos)

    all_inputs, all_outputs = [], []
    print(f"Generating {n_train + n_test} samples...")
    for i in tqdm(range(n_train + n_test), desc="Generating samples"):
        coeffs = rng.standard_normal(n_terms)
        for idx, (k, m, _) in enumerate(terms):
            coeffs[idx] /= (1.0 + 0.5 * (k + m))

        u = coeffs @ basis_vals
        f = -(coeffs @ basis_laps)
        u_scale = np.max(np.abs(u)) + 1e-12
        u, f = u / u_scale, f / u_scale

        all_inputs.append(torch.from_numpy(f).float().unsqueeze(-1))
        all_outputs.append(torch.from_numpy(u).float().unsqueeze(-1))

        split_dir = train_dir if i < n_train else test_dir
        idx_in_split = i if i < n_train else i - n_train
        np.savetxt(os.path.join(split_dir, f"sample_{idx_in_split:04d}.txt"), 
                   np.column_stack([x_irreg, y_irreg, f, u]), header="x y f u", fmt="%.10f", comments="")

    # Save as .pt for training script
    torch.save({
        'coords': coords,
        'inputs': torch.stack(all_inputs),
        'outputs': torch.stack(all_outputs),
        'latent_grid': latent_grid
    }, pt_path)
    print(f"Saved dataset to {pt_path}")

    # Verification
    theta_check = np.linspace(0, 2 * pi, 200)
    u_inner = sum(coeffs[idx] * evaluate_basis(np.full(200, r_in), theta_check, k, m, r_in, r_out, is_cos) 
                  for idx, (k, m, is_cos) in enumerate(terms)) / u_scale
    u_outer = sum(coeffs[idx] * evaluate_basis(np.full(200, r_out), theta_check, k, m, r_in, r_out, is_cos) 
                  for idx, (k, m, is_cos) in enumerate(terms)) / u_scale
    print(f"BC Check: inner {np.max(np.abs(u_inner)):.2e}, outer {np.max(np.abs(u_outer)):.2e}")

    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, si in enumerate([0, n_train // 2, n_train - 1]):
        d = np.loadtxt(os.path.join(train_dir, f"sample_{si:04d}.txt"), skiprows=1)
        xv, yv, fv, uv = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        for row, (val, title, cmap) in enumerate([(fv, "Source f", "inferno"), (uv, "Solution u", "viridis")]):
            sc = axes[row, col].scatter(xv, yv, c=val, s=1, cmap=cmap)
            axes[row, col].set_title(f"{title} (s{si})")
            axes[row, col].set_aspect("equal")
            for rad in [r_in, r_out]:
                axes[row, col].add_patch(plt.Circle((0, 0), rad, fill=False, color="white", linewidth=0.5, linestyle="--"))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_visualisation.png"), dpi=100)
    plt.close()
    return out_dir

if __name__ == "__main__":
    generate_poisson_donut(n_train=200, n_test=30)

