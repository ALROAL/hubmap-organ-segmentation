from ..PATHS import CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
from hubmap.data.create_dataloaders import prepare_test_loader, prepare_val_loader
from hubmap.models.models import load_model
from hubmap.models.predict_model import predict_batch
import torch
import matplotlib.pyplot as plt


def visualize_random_segmentations(model_type, model_path, dataset="val", val_fold=0, n=5, threshold=0.5, batch_size=1, device=CFG["device"]):

    if dataset=="test":
        data_loader = prepare_test_loader(batch_size, shuffle=True)
    elif dataset=="val":
        data_loader = prepare_val_loader(val_fold, batch_size, val_shuffle=True)

    n_batches = int(n / data_loader.batch_size) + (n % data_loader.batch_size)

    all_images = []
    all_segmented_images = []
    all_masks = []
    for _ in range(n_batches):
        images, masks, H, W = next(iter(data_loader))
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)

        segmented_images = predict_batch(model_type, model_path, images, H, W, threshold, device=CFG["device"])

        all_images.append(images)
        all_segmented_images.append(segmented_images)
        all_masks.append(masks)
    
    all_images = torch.cat(all_images[:n]).permute(0,2,3,1).cpu().detach()
    all_segmented_images = torch.cat(all_segmented_images[:n]).permute(0,2,3,1).cpu().detach()
    all_masks = torch.cat(all_masks[:n]).permute(0,2,3,1).cpu().detach()

    f, axes = plt.subplots(n, 3, figsize=(15, 15))
    axes[0,0].set_title("Image")
    axes[0,1].set_title("Ground Truth")
    axes[0,2].set_title("Segmentation")

    for i in range(n):
        axes[i,0].imshow(all_images[i])
        axes[i,1].imshow(all_masks[i])
        axes[i,2].imshow(all_segmented_images[i])