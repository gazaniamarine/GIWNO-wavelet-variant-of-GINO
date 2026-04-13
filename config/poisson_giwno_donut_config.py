import torch

# GIWNO (Wavelet Variant) Configuration
config = {
    "data": {
        "path": "/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/data/poisson_donut.pt",
        "n_samples": 1000,
        "batch_size": 16,
        "train_test_split": 0.8
    },
    "model": {
        "type": "giwno",
        "in_channels": 4,
        "out_channels": 1,
        "gno_coord_dim": 2,
        "wno": {
            "n_modes": (16, 16),
            "hidden_channels": 128,
            "n_layers": 4,
            "in_channels": 4,
            "level": 2,
            "wavelet": 'haar'
        },
        "gno": {
            "in_radius": 0.05,
            "out_radius": 0.05,
            "use_open3d": False
        }
    },
    "train": {
        "epochs": 500,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler_t_max": 500,
        "loss_type": "rel_l2"
    },
    "output": {
        "save_dir": "/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/outputs/giwno",
        "model_name": "giwno_wavelet.pt"
    }
}
