import torch

# GINO (FNO Baseline) Configuration
config = {
    "data": {
        "path": "/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/data/poisson_donut.pt",
        "n_samples": 1000,
        "batch_size": 16,
        "train_test_split": 0.8
    },
    "model": {
        "type": "gino",
        "in_channels": 1,
        "out_channels": 1,
        "gno_coord_dim": 2,
        "fno": {
            "n_modes": (16, 16),
            "hidden_channels": 64,
            "n_layers": 4,
            "in_channels": 16,
        },
        "gno": {
            "in_radius": 0.05,
            "out_radius": 0.05,
            "use_open3d": False
        }
    },
    "train": {
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "scheduler_t_max": 100,
        "loss_type": "rel_l2"
    },
    "output": {
        "save_dir": "/home/gazania/zania_folder/GIWNO-wavelet-variant-of-GINO/outputs",
        "model_name": "gino_fno_baseline.pt"
    }
}
