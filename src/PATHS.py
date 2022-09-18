from config import CFG
from pathlib import Path

DATA_PATH = Path(CFG.data_path)

IMAGES_PATH = DATA_PATH / "images"
ANNOTATIONS_PATH = DATA_PATH / "annotations"
DATA_CSV_PATH = DATA_PATH / "data.csv"

TRAIN_IMAGES_PATH = DATA_PATH / f"train_images_{CFG.img_size}x{CFG.img_size}"
TRAIN_MASKS_PATH = DATA_PATH / f"train_masks_{CFG.img_size}x{CFG.img_size}"
TRAIN_CSV_PATH = DATA_PATH / "train.csv"

TEST_IMAGES_PATH = DATA_PATH / "test_images"
TEST_MASKS_PATH = DATA_PATH / "test_masks"
TEST_CSV_PATH = DATA_PATH / "test.csv"

MODEL_PATH = Path(CFG.model_path) / CFG.model / "model.bin"


