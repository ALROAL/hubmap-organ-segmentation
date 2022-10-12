from ..PATHS import CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
import torch
import numpy as np
from hubmap.data import prepare_val_loader, prepare_test_loader
from hubmap.models import load_model
from segmentation_models_pytorch.losses import DiceLoss
import cv2

def evaluate_model(model_type, model_path, batch_size=1, dataset="test", val_fold=0, device=CFG["device"]):

    model = load_model(model_type, model_path, inference=True, device=device)

    if dataset=="test":
        data_loader = prepare_test_loader()
    else:
        data_loader = prepare_val_loader(val_fold)

    dice_score_sum = 0
    n_samples=0
    for images, masks, H, W in data_loader:
        H = int(H)
        W = int(W)
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)
        n_samples += batch_size

        segmented_batch = model(images).permute(0,2,3,1).cpu().detach().numpy()
        resized_segmented_batch = torch.tensor([])
        for segmented_image in segmented_batch:

            segmented_image = cv2.resize(segmented_image, (H, W))
            segmented_image = np.expand_dims(segmented_image, (0,1))
            segmented_image = torch.tensor(segmented_image).to(device, dtype=torch.float)
            resized_segmented_batch = torch.cat((resized_segmented_batch, segmented_image))

        dice_score_sum += (1 - DiceLoss(mode='binary', from_logits=False)(resized_segmented_batch, masks))*batch_size

    dice_score = dice_score_sum / n_samples

    return dice_score

