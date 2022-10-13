from ..PATHS import CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
import torch
import numpy as np
from hubmap.data import prepare_val_loader, prepare_test_loader
from hubmap.models import load_model
from hubmap.models.predict_model import predict_batch
from segmentation_models_pytorch.losses import DiceLoss

@torch.no_grad()
def evaluate_model(model_type, model_path, threshold=0.5, batch_size=1, dataset="test", val_fold=0, device=CFG["device"]):

    if dataset=="test":
        data_loader = prepare_test_loader(batch_size)
    else:
        data_loader = prepare_val_loader(val_fold, batch_size)

    dice_score_sum = 0
    n_samples=0
    for images, masks, H, W in data_loader:

        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)
        n_samples += batch_size

        resized_segmented_batch = predict_batch(model_type, model_path, images, H, W, threshold, device)
        dice_score_sum += (1 - DiceLoss(mode='binary', from_logits=False)(resized_segmented_batch, masks))*batch_size

    dice_score = dice_score_sum / n_samples

    return dice_score

