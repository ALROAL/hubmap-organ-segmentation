import wandb
from .train_model import train, test
from ..PATHS import CONFIG_JSON_PATH, MODELS_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)

def model_pipeline():
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    wandb.login()
    config = dict(
        model = CFG["model"],
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        loss=CFG["loss"],
        optimizer=CFG["optimizer"],
        learning_rate=CFG["lr"],
        scheduler=CFG["scheduler"]
    )
    train(config)

