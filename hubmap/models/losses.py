import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

#Loss function
def bce_loss(y_pred, y_true):
    loss = nn.BCELoss()(y_pred, y_true)
    loss = {"BCE": loss}
    return loss

def dice_coef(y_pred, y_true, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def soft_dice_loss(y_pred, y_true, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    dice_loss = 1-dice
    loss = {"SoftDice": dice_loss}
    return loss

def bce_soft_dice_loss(y_pred, y_true):
    dice_loss = soft_dice_loss(y_pred, y_true)["SoftDice"]
    bce_l = bce_loss(y_pred, y_true)["BCE"]
    bce_dice_loss = (dice_loss + bce_l)/2
    loss = {"BCE": bce_l, "SoftDice": dice_loss, "BCE+SoftDice": bce_dice_loss}
    return loss

def iou_coef(y_pred, y_true, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def JaccardLoss(y_pred, y_true):
    loss = smp.losses.JaccardLoss(mode='binary')(y_pred, y_true)
    return {"Jaccard": loss}
def DiceLoss(y_pred, y_true):
    loss = smp.losses.DiceLoss(mode='binary')(y_pred, y_true)
    return {"Dice": loss}
def BCELoss(y_pred, y_true):
    loss = nn.BCELoss()(y_pred, y_true)
    return {"BCE": loss}
def LovaszLoss(y_pred, y_true):
    loss = smp.losses.LovaszLoss(mode='binary', per_image=False)(y_pred, y_true)
    return {"Lovasz": loss}
def TverskyLoss(y_pred, y_true):
    loss = smp.losses.TverskyLoss(mode='binary', log_loss=False)(y_pred, y_true)
    return {"Tversky": loss}

def BCEDiceLoss(y_pred, y_true):
    bce_loss = nn.BCEWithLogitsLoss()(y_pred, y_true)
    dice_loss = smp.losses.DiceLoss(mode='binary')(y_pred, y_true)
    loss = (bce_loss + dice_loss)/2.
    return {"BCE": bce_loss, "Dice": dice_loss, "BCEDice": loss}