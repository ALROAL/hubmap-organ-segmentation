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

    all_segmented_images =  []
    all_masks = []
    for images, masks in data_loader:
        segmented_batch = predict_with_smooth_windowing(model_type, model_path, images, window_size=512, subdivisions=2, nb_classes=1, device=device)
        all_segmented_images.append(segmented_batch)
        all_masks.append(masks)
    
    all_segmented_images = np.array(all_segmented_images)
    all_segmented_images = torch.tensor(all_segmented_images)

    all_masks = np.array(all_masks)
    all_masks = torch.tensor(all_masks)

    dice_loss = DiceLoss(mode='binary')(all_segmented_images, all_masks)

    return 1-dice_loss

