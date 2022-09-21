import json
from pathlib import Path
import os

CONFIG_JSON_PATH = str(Path(os.getcwd()) / "src/config.json")
with open(CONFIG_JSON_PATH) as f:
  CFG = json.load(f)

DATA_PATH = Path(CFG["data_path"]) if CFG["data_path"] else Path(os.sep.join(os.getcwd().split(os.sep)[:-1] + ["data"]))

IMAGES_PATH = DATA_PATH / "images"
ANNOTATIONS_PATH = DATA_PATH / "annotations"
DATA_CSV_PATH = DATA_PATH / "data.csv"

TRAIN_IMAGES_PATH = DATA_PATH / f"train_images_{CFG['img_size']}x{CFG['img_size']}"
TRAIN_MASKS_PATH = DATA_PATH / f"train_masks_{CFG['img_size']}x{CFG['img_size']}"
TRAIN_CSV_PATH = DATA_PATH / "train.csv"

TEST_IMAGES_PATH = DATA_PATH / "test_images"
TEST_MASKS_PATH = DATA_PATH / "test_masks"
TEST_CSV_PATH = DATA_PATH / "test.csv"

MODEL_PATH = (Path(CFG["model_path"]) if CFG["model_path"] else Path(os.sep.join(os.getcwd().split(os.sep)[:-1] + ["models"]))) / CFG["model"] / "model.bin"



