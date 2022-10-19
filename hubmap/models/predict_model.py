from ..PATHS import CONFIG_JSON_PATH
import numpy as np
import torch
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
from .models import load_model
import cv2

def predict_batch(model_type, model_path, images, H, W, threshold=0.5, device=CFG["device"]):

    model = load_model(model_type, model_path, inference=True, device=device)

    images = images.to(device, dtype=torch.float)

    segmented_batch = model(images).permute(0,2,3,1).cpu().detach().numpy()
    resized_segmented_batch = []
    for i,segmented_image in enumerate(segmented_batch):
        segmented_image = cv2.resize(segmented_image, (int(H[i]), int(W[i])))
        segmented_image = np.expand_dims(segmented_image, (0,1))
        if threshold:
            segmented_image = torch.tensor(segmented_image>threshold).to(device ,dtype=torch.float)
        else:
            segmented_image = torch.tensor(segmented_image).to(device, dtype=torch.float)
        resized_segmented_batch.append(segmented_image)

    return resized_segmented_batch

