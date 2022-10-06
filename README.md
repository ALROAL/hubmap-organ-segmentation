U-Net: Semantic segmentation with PyTorch for Kaggle's competition: HuBMAP + HPA - Hacking the Human Body
==============================

The goal of this competition is to identify the locations of each functional tissue unit (FTU) in biopsy slides from several different organs. The underlying data includes imagery from different sources prepared with different protocols at a variety of resolutions, reflecting typical challenges for working with medical data.

- [Quick start](#quick-start)
- [Description](#description)
- [Usage](#usage)
- [Weights & Biases](#weights--biases)

## Quick start

1. Install dependencies
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu116/torch_stable.html
```
2. Download data directly from Kaggle's competition [HuBMAP + HPA - Hacking the Human Body](https://www.kaggle.com/competitions/hubmap-organ-segmentation/data) or using DVC (this will create a ```data``` folder in the project's directory).
```bash
dvc pull
```
Note: The DVC option might not be available due to the removal of the data from Google drive.

## Description
The project implements the UNet architecture from [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
## Usage

```console
> python main.py -h
usage: main.py [-h] [--data_path DATA_PATH] [--num_classes NUM_CLASSES] [--img_size IMG_SIZE] [--model_path MODEL_PATH] [--seed SEED] [--test_size TEST_SIZE] [--n_folds N_FOLDS] [--model MODEL] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--loss LOSS] [--optimizer OPTIMIZER] [--lr LR] [--weight_decay WEIGHT_DECAY] [--scheduler SCHEDULER] [--n_accumulate N_ACCUMULATE] [--device DEVICE]
               [--num-workers NUM_WORKERS]

Creates the data and train a model on HuBMAP + HPA dataset

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Specify a diferent data path from the default 'path/to/hubmap-organ-segmentation/data'
  --num_classes NUM_CLASSES
                        Number of different classes to segment
  --img_size IMG_SIZE   Image size to use
  --model_path MODEL_PATH
                        Specify a diferent model path from the default 'path/to/hubmap-organ-segmentation/models'
  --seed SEED           Seed for random generation
  --test_size TEST_SIZE
                        Relative size (0-1) of the test set
  --n_folds N_FOLDS     Number of folds for cross validation
  --model MODEL         Model to train
  --epochs EPOCHS       Number of epochs to train
  --batch_size BATCH_SIZE
                        Batch size to train
  --loss LOSS           Loss function to train
  --optimizer OPTIMIZER
                        Optimizer to train
  --lr LR               Initial learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay for the learning rate
  --scheduler SCHEDULER
                        Scheduler for the learning rate
  --n_accumulate N_ACCUMULATE
                        Number of batches to accumulate for gradients
  --device DEVICE       Device (CUDA or cpu)
  --num-workers NUM_WORKERS
                        Number of threads use in loading data
```

## Weights & Biases
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/). Training and validation loss curves are logged to the platform.
When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.
