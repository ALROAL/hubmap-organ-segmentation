from ..PATHS import IMAGES_PATH, MASKS_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH, CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
import torch
import albumentations as A
import pandas as pd
import numpy as np

from .utils import *


class HuBMAP_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labeled=True, transforms=None, evaluation=False, evaluation_transforms=None):
        self.df = df
        self.labeled = labeled
        self.transforms = transforms
        self.evaluation_transforms = evaluation_transforms
        self.evaluation = evaluation
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.loc[index, 'image_path']
        ext = img_path.split(".")[-1]
        if ext=="tiff":
            img = read_tiff(img_path)
        else:
            img = read_img(img_path)
        
        if self.labeled:
            mask_path = self.df.loc[index, 'mask_path']
            mask = read_mask(mask_path)
            
            if self.transforms:
                data = self.transforms(image=img, mask=mask)
                img  = data['image']
            if self.evaluation_transforms:
                data = self.evaluation_transforms(image=img, mask=mask)

            mask  = data['mask']
            
            mask = np.expand_dims(mask, axis=0)
            img = np.transpose(img, (2, 0, 1))
#             mask = np.transpose(mask, (2, 0, 1))

            if self.evaluation:
                return torch.tensor(img), torch.tensor(mask), self.df.loc[index, 'img_height'], self.df.loc[index, 'img_width']
            else:
                return torch.tensor(img), torch.tensor(mask)
        
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
                
            img = np.transpose(img, (2, 0, 1))
            
            return torch.tensor(img)

data_transforms = {
    "train": A.Compose([
        A.Resize(CFG["img_size"], CFG["img_size"], interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.OneOf([
            A.HueSaturationValue(10,15,10),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.4),
        A.ToFloat(255),
    ]),
    
    "valid": A.Compose([
        A.Resize(CFG["img_size"], CFG["img_size"], interpolation=cv2.INTER_LINEAR),
        A.ToFloat(255),
        ]),

    "test": A.Compose([A.ToFloat(255)])
}

def prepare_train_loaders(fold, val_shuffle=False):
    
    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df = df[df["fold"]!=fold].reset_index(drop=True)
    valid_df = df[df["fold"]==fold].reset_index(drop=True)

    train_dataset = HuBMAP_Dataset(train_df, transforms=data_transforms['train'])
    valid_dataset = HuBMAP_Dataset(valid_df, transforms=data_transforms['valid'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG["batch_size"], 
        num_workers=CFG["num_workers"], shuffle=True, drop_last=False)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], shuffle=val_shuffle)
    
    return train_loader, valid_loader

def prepare_val_loader(fold, batch_size=1, shuffle=False):

    df = pd.read_csv(TRAIN_CSV_PATH)
    valid_df = df[df["fold"]==fold].reset_index(drop=True)

    valid_dataset = HuBMAP_Dataset(valid_df, transforms=data_transforms['valid'], evaluation=True, evaluation_transforms=data_transforms['test'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
        num_workers=CFG["num_workers"], shuffle=shuffle, drop_last=False)

    return valid_loader

def prepare_test_loader(batch_size=1, shuffle=False):

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_dataset = HuBMAP_Dataset(test_df, transforms=data_transforms['valid'], evaluation=True, evaluation_transforms=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
        num_workers=CFG["num_workers"], shuffle=shuffle, drop_last=False)

    return test_loader