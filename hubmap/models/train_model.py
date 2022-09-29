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
    dice_loss = 1-dice
    loss = {"SoftDice": dice_loss}
    return loss

def bce_soft_dice_loss(y_pred, y_true):
    dice_loss = soft_dice_loss(y_pred, y_true)
    bce_loss = nn.BCELoss()(y_pred, y_true)
    bce_dice_loss = (dice_loss + bce_loss)/2
    loss = {"BCE": bce_loss, "SoftDice": dice_loss, "BCE+SoftDice": bce_dice_loss}
    return loss

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
def get_scheduler(optimizer):
    
    T_max = int(CFG["epochs"]/3)
    if CFG["scheduler"] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    elif CFG["scheduler"] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, threshold=0.0001, min_lr=1e-6)

    elif CFG["scheduler"] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        
    return scheduler


#Train functions
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler):
    wandb.watch(model, log=None)
    model.train()
    losses_sum = {}
    epoch_losses = {}
    n_samples = 0
    first = True
    for images, masks in dataloader:

        images = images.to(CFG["device"], dtype=torch.float)
        masks  = masks.to(CFG["device"], dtype=torch.float)

        batch_size = images.size(0)
        n_samples += batch_size
        
        y_pred = model(images)
        losses = criterion(y_pred, masks)

        if first:
            for k, v in losses.items():
                losses_sum[k] = v.item()*batch_size
        else:
            for k, v in losses.items():
                losses_sum[k] += v.item()*batch_size

        loss = losses[CFG["loss"]]
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    if scheduler is not None:
        scheduler.step()
    
    for k, v in losses_sum:
        epoch_losses[k] = v / n_samples
    torch.cuda.empty_cache()
    
    return epoch_losses

@torch.no_grad()
def valid_one_epoch(model, dataloader):
    model.eval()

    losses_sum = {}
    epoch_losses = {}
    n_samples = 0

    criterion = get_loss()
    first = True
    for images, masks in dataloader:        
        images = images.to(CFG["device"], dtype=torch.float)
        masks  = masks.to(CFG["device"], dtype=torch.float)

        batch_size = images.size(0)
        n_samples += batch_size
        
        y_pred = model(images)
        losses = criterion(y_pred, masks)
        if first:
            for k, v in losses.items():
                losses_sum[k] = v.item()*batch_size
        else:
            for k, v in losses.items():
                losses_sum[k] += v.item()*batch_size
        
    for k, v in losses_sum:
        epoch_losses[k] = v / n_samples
    torch.cuda.empty_cache()
    
    return epoch_losses


def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler):
    # To automatically log gradients
    wandb.watch(model, log=None)

    best_loss = np.inf
    for epoch in range(CFG["epochs"]):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss = valid_one_epoch(model, val_loader)

        logging_dict = {}
        for k, v in train_loss.items():
            logging_dict[f"train_{k}"] = v
        for k, v in val_loss.items():
            logging_dict[f"val_{k}"] = v

        wandb.log(logging_dict)
        # deep copy the model weights
        if val_loss < best_loss:
            best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def train(config):

    for fold in range(CFG["n_folds"]):

        model = build_model()
        model.apply(initialize_weights)
        train_loader, val_loader = prepare_train_loaders(fold)
        criterion = get_loss()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)

        wandb.init(project="hubmap-organ-segmentation", group=CFG["model"], config=config, job_type='train', name=f'fold_{fold}')

        model = run_training(model, train_loader, val_loader, criterion, optimizer, scheduler)
        model_name = f'model_fold_{fold}.pth'
        save_model(model, model_name)

        wandb.join()

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