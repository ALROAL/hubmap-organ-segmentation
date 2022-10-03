from ..PATHS import CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
from .models import load_model

def segment_images(model_type, path, images, device=CFG["device"]):
    model = load_model(model_type, path, device)
    segmented_images = model(images)
    return segmented_images