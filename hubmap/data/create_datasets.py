# -*- coding: utf-8 -*-
from ..PATHS import *
import json
with open(CONFIG_JSON_PATH) as f:
  CFG = json.load(f)
import cv2
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from .utils import *


def create_datasets():

    MASKS_PATH.mkdir(parents=True, exist_ok=True)

    #Load images metadata from csv
    data = pd.read_csv(DATA_CSV_PATH)
    for _, row in data.iterrows():
        mask = rle_decode(row.rle, (row.img_height, row.img_width))
        mask_path = MASKS_PATH / (str(row.id) + ".png")
        cv2.imwrite(str(mask_path), mask)

    data = data[["id", "organ", "data_source", "img_height", "img_width"]]

    #Create train/test split
    train_data, test_data = train_test_split(data, test_size=CFG["test_size"], stratify=data["organ"], random_state=CFG["seed"])
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data["image_path"] = str(IMAGES_PATH) + "/" + train_data["id"].apply(str) + ".tiff"
    train_data["mask_path"] = str(MASKS_PATH) + "/" + train_data["id"].apply(str) + ".png"
    test_data["image_path"] = str(IMAGES_PATH) + "/" + test_data["id"].apply(str) + ".tiff"
    test_data["mask_path"] = str(MASKS_PATH) + "/" + test_data["id"].apply(str) + ".png"

    kf = GroupKFold(n_splits=CFG["n_folds"])
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data, groups=train_data['id'])):
        train_data.loc[val_idx, 'fold'] = fold

    test_data.to_csv(TEST_CSV_PATH, index=False)
    train_data.to_csv(TRAIN_CSV_PATH, index=False)

