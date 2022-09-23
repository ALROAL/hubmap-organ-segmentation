import torch
import os

CFG = {
    "data_path": "", #specify a diferent data path from the default "path/to/hubmap-organ-segmentation/data"
    "num_classes": 1,
    "img_size": 256,
    "model_path": "", #specify a diferent model path from the default "path/to/hubmap-organ-segmentation/models"
    "seed": 0,
    "test_size": 0.2,
    "n_folds": 4,
    "model": "UNet",
    "epochs": 100,
    "batch_size": 64,
    "loss": "BCE+SoftDice",
    "optimizer": 'Adam',
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "scheduler": 'CosineAnnealingLR', #['CosineAnnealingLR', 'ReduceLROnPlateau', 'ExponentialLR']
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "num_workers": os.cpu_count()
}