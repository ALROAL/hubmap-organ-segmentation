import json
from pathlib import Path
import os

SRC_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_JSON_PATH = str(SRC_PATH / "config.json")
with open(CONFIG_JSON_PATH) as f:
  CFG = json.load(f)

DATA_PATH = Path(CFG["data_path"]) if CFG["data_path"] else Path(os.sep.join(str(SRC_PATH).split(os.sep)[:-1] + ["data"]))

IMAGES_PATH = DATA_PATH / "images"
ANNOTATIONS_PATH = DATA_PATH / "annotations"
MASKS_PATH = DATA_PATH / "masks"
DATA_CSV_PATH = DATA_PATH / "data.csv"

TRAIN_CSV_PATH = DATA_PATH / "train.csv"

TEST_CSV_PATH = DATA_PATH / "test.csv"
MODELS_PATH = Path(os.sep.join(str(SRC_PATH).split(os.sep)[:-1] + ["models"]))
MODEL_PATH = Path(CFG["model_path"]) if CFG["model_path"] else MODELS_PATH / CFG["model"]



