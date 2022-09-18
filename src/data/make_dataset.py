# -*- coding: utf-8 -*-
from config import CFG
from PATHS import *

import cv2
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
import shutil
from utils import *

def resize_with_padding(img, width, height):
    H = img.shape[0]
    W = img.shape[1]
    delta_w = width - W
    delta_h = height - H
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


if __name__ == "__main__":
    #Create folder to store cropped images
    TRAIN_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_MASKS_PATH.mkdir(parents=True, exist_ok=True)
    TEST_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    TEST_MASKS_PATH.mkdir(parents=True, exist_ok=True)

    #Load images metadata from csv
    data = pd.read_csv(DATA_CSV_PATH)

    #Create train/test split
    train_data, test_data = train_test_split(data, test_size=CFG.test_size, stratify=data["organ"], random_state=CFG.seed)

    id_list = []
    id_2_list = []
    img_path_list = []
    mask_path_list = []
    for _, row in train_data.iterrows():

        #Read image and its mask
        img = read_tiff(IMAGES_PATH / (str(row.id) + ".tiff"))
        mask = rle_decode(row.rle, (row.img_height, row.img_width))

        #Compute number of crops
        n_rows = int(row.img_height // CFG.img_size)+1
        n_cols = int(row.img_width // CFG.img_size)+1

        resize_height = n_rows * CFG.img_size
        resize_width = n_cols * CFG.img_size

        #Add padding for crops to be equal-sized
        img = resize_with_padding(img, resize_width, resize_height)
        mask = resize_with_padding(mask, resize_width, resize_height)

        for i in range(n_rows):
            for j in range(n_cols):

                    mask_path = TRAIN_MASKS_PATH / (str(row.id) + f"_{i}_{j}.png")
                    mask_crop = mask[i*CFG.img_size:(i+1)*CFG.img_size, j*CFG.img_size:(j+1)*CFG.img_size]

                    if mask_crop.sum() <= 0:
                        continue

                    img_path = TRAIN_IMAGES_PATH / (str(row.id) + f"_{i}_{j}.png")
                    img_crop = img[i*CFG.img_size:(i+1)*CFG.img_size, j*CFG.img_size:(j+1)*CFG.img_size]
                    
                    cv2.imwrite(str(img_path), img_crop)
                    cv2.imwrite(str(mask_path), mask_crop)

                    id_list.append(row.id)
                    id_2_list.append(str(row.id) + f"_{i}_{j}")
                    img_path_list.append(img_path)
                    mask_path_list.append(mask_path)

    df = pd.DataFrame({"id":id_list, "id_2":id_2_list, "image_path":img_path_list, "mask_path":mask_path_list})

    kf = GroupKFold(n_splits=CFG.n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, groups=df['id'])):
        df.loc[val_idx, 'fold'] = fold

    df.to_csv(TRAIN_CSV_PATH, index=False)

    for _, row in test_data.iterrows():

        #Move image to test folder
        shutil.copy(IMAGES_PATH / (str(row.id) + ".tiff"), TEST_IMAGES_PATH /(str(row.id) + ".tiff"))
        #Read and save mask
        mask = rle_decode(row.rle, (row.img_height, row.img_width))
        cv2.imwrite(str(TEST_MASKS_PATH / (str(row.id) + ".png")), mask)

    test_data["image_path"] = TEST_IMAGES_PATH / (str(test_data["id"]) + ".tiff")
    test_data.to_csv(TEST_CSV_PATH, index=False)





