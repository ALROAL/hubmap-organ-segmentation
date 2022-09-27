import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

from ..PATHS import CONFIG_JSON_PATH
import json
with open(CONFIG_JSON_PATH) as f:
  CFG = json.load(f)
from .models import build_model, save_model
from ..data.create_dataloaders import prepare_train_loaders, prepare_test_loader
import wandb

#Weight initialization
def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.xavier_normal_(m.weight.data)

#Loss function
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

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
    return 1-dice

def bce_soft_dice_loss(y_pred, y_true):
    return (soft_dice_loss(y_pred, y_true) + nn.BCELoss()(y_pred, y_true))/2

def iou_coef(y_pred, y_true, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def get_loss():
    if CFG["loss"] == "BCE+SoftDice":
        return bce_soft_dice_loss
    elif CFG["loss"] == "SoftDice":
        return SoftDiceLoss()
    elif CFG["loss"] == "BCE":
        return nn.BCELoss()

#Optimizer
def get_optimizer(model):
    if CFG["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    
    elif CFG["optimizer"] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        
    return optimizer

#Scheduler
def get_scheduler(optimizer, train_loader):
    
    num_steps = len(train_loader)
    
    if CFG["scheduler"] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)

    elif CFG["scheduler"] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, threshold=0.0001, min_lr=1e-6)

    elif CFG["scheduler"] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        
    return scheduler


#Train functions
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler):
    wandb.watch(model, log=None)
    model.train()
    loss_sum = 0
    n_samples = 0

    for images, masks in dataloader:

        images = images.to(CFG["device"], dtype=torch.float)
        masks  = masks.to(CFG["device"], dtype=torch.float)


        batch_size = images.size(0)
        n_samples += batch_size
        
        y_pred = model(images)
        loss = criterion(y_pred, masks)
        loss_sum += loss.item()*batch_size

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    if scheduler is not None:
        scheduler.step()
    
    epoch_loss = loss_sum / n_samples
    torch.cuda.empty_cache()
    
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader):
    model.eval()
    
    loss_sum = 0
    n_samples = 0
    criterion = get_loss()
    
    for images, masks in dataloader:        
        images = images.to(CFG["device"], dtype=torch.float)
        masks  = masks.to(CFG["device"], dtype=torch.float)

        batch_size = images.size(0)
        n_samples += batch_size
        
        y_pred = model(images)
        loss = criterion(y_pred, masks)
        loss_sum += loss.item()*batch_size
        
    epoch_loss = loss_sum / n_samples
    torch.cuda.empty_cache()
    
    return epoch_loss


def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler):
    # To automatically log gradients
    wandb.watch(model, log=None)

    best_loss = np.inf
    for epoch in range(CFG["epochs"]):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        print(f"Train loss {train_loss}")
        val_loss = valid_one_epoch(model, val_loader)
        print(f"Val loss {val_loss}")
        wandb.log(
            {"epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss}
            )

        # deep copy the model weights
        if val_loss < best_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(model)

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model(model)

    wandb.join()

    return model

def train():

    for fold in range(CFG["n_folds"]):
        model = build_model()
        model.apply(initialize_weights)
        train_loader, val_loader = prepare_train_loaders(fold)
        criterion = get_loss()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, train_loader)
        model = run_training(model, train_loader, val_loader, criterion, optimizer, scheduler)
    return model

@torch.no_grad()
def test(model):
    test_loader = prepare_test_loader()
    model.eval()
    loss = get_loss()
    
    iou_sum = 0
    loss_sum = 0
    n_samples = 0
    
    for images, masks in test_loader:        
        images = images.to(CFG["device"], dtype=torch.float)
        masks  = masks.to(CFG["device"], dtype=torch.float)

        batch_size = images.size(0)
        n_samples += batch_size
        
        y_pred = model(images)
        iou_sum += iou_coef(y_pred, masks)
        loss_sum += loss(y_pred, masks)
        
    test_iou_score = iou_sum / n_samples
    test_loss = loss_sum / n_samples
    torch.cuda.empty_cache()
    return test_loss, test_iou_score