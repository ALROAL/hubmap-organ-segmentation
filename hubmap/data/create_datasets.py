# -*- coding: utf-8 -*-
from ..PATHS import *
import json
with open(CONFIG_JSON_PATH) as f:
  CFG = json.load(f)
import cv2
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from .utils import *


def create_crop_datasets():

    #Create folder to store cropped images
    MASKS_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_MASKS_PATH.mkdir(parents=True, exist_ok=True)

    #Load images metadata from csv
    data = pd.read_csv(DATA_CSV_PATH)

    #Create train/test split
    train_data, test_data = train_test_split(data, test_size=CFG["test_size"], stratify=data["organ"], random_state=CFG["seed"])

    id_list = []
    id_2_list = []
    img_path_list = []
    mask_path_list = []
    for _, row in train_data.iterrows():

        #Read image and its mask
        img = read_tiff(IMAGES_PATH / (str(row.id) + ".tiff"))
        mask = rle_decode(row.rle, (row.img_height, row.img_width))

        mask_path = MASKS_PATH / (str(row.id) + ".png")
        cv2.imwrite(str(mask_path), mask)

        #Compute number of crops
        n_rows = int(row.img_height // CFG["img_size"])+1
        n_cols = int(row.img_width // CFG["img_size"])+1

        resize_height = n_rows * CFG["img_size"]
        resize_width = n_cols * CFG["img_size"]

        #Add padding for crops to be equal-sized
        img = resize_with_padding(img, resize_width, resize_height)
        mask = resize_with_padding(mask, resize_width, resize_height)

        for i in range(n_rows):
            for j in range(n_cols):

                mask_path = str(TRAIN_MASKS_PATH / (str(row.id) + f"_{i}_{j}.png"))
                mask_crop = mask[i*CFG["img_size"]:(i+1)*CFG["img_size"], j*CFG["img_size"]:(j+1)*CFG["img_size"]]

                img_path = str(TRAIN_IMAGES_PATH / (str(row.id) + f"_{i}_{j}.png"))
                img_crop = img[i*CFG["img_size"]:(i+1)*CFG["img_size"], j*CFG["img_size"]:(j+1)*CFG["img_size"]]
                
                cv2.imwrite(str(img_path), img_crop)
                cv2.imwrite(str(mask_path), mask_crop)

                id_list.append(row.id)
                id_2_list.append(str(row.id) + f"_{i}_{j}")
                img_path_list.append(img_path)
                mask_path_list.append(mask_path)

    df = pd.DataFrame({"id":id_list, "id_2":id_2_list, "image_path":img_path_list, "mask_path":mask_path_list})

    kf = GroupKFold(n_splits=CFG["n_folds"])
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, groups=df['id'])):
        df.loc[val_idx, 'fold'] = fold

    df.to_csv(TRAIN_CSV_PATH, index=False)

    test_data.drop("rle", axis=1, inplace=True)
    test_data["image_path"] = str(IMAGES_PATH) + "/" + test_data["id"].apply(str) + ".tiff"
    test_data["mask_path"] = str(MASKS_PATH) + "/" + test_data["id"].apply(str) + ".png"
    
    test_data.to_csv(TEST_CSV_PATH, index=False)



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

