from ..PATHS import CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
import numpy as np
import torch
from hubmap.data.create_dataloaders import prepare_val_loader, prepare_test_loader
from hubmap.models.predict_model import predict_with_smooth_windowing
from segmentation_models_pytorch.losses import DiceLoss

def evaluate_model(model_type, model_path, dataset="test", val_fold=0, device=CFG["device"]):

    if dataset=="test":
        data_loader = prepare_test_loader()
    else:
        data_loader = prepare_val_loader(val_fold)

    all_segmented_images =  torch.tensor([])
    all_masks = torch.tensor([])
    for images, masks in data_loader:
        segmented_batch = predict_with_smooth_windowing(model_type, model_path, images, window_size=512, subdivisions=2, nb_classes=1, device=device)
        all_segmented_images = torch.cat((all_segmented_images, segmented_batch), 0)
        all_masks = torch.cat((all_masks, masks), 0)

    dice_loss = DiceLoss(mode='binary')(all_segmented_images, all_masks)

    return 1-dice_loss

