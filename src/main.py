import argparse
from data.create_datasets import create_datasets
from models.pipeline import model_pipeline
import torch
import os

class CFG:
    data_path = "" #specify a diferent data path from the default "path/to/hubmap-organ-segmentation/data"
    num_classes = 1
    img_size = 512
    model_path = "" #specify a diferent model path from the default "path/to/hubmap-organ-segmentation/models"
    seed = 0
    test_size = 0.2
    n_folds = 5
    model = "UNet"
    epochs = 20
    batch_size = 2
    loss = "BCE+SoftDice"
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 1e-6
    scheduler = 'CosineAnnealingLR' #['CosineAnnealingLR', 'ReduceLROnPlateau', 'ExponentialLR']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates the data and train a model on HuBMAP + HPA dataset"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Specify a diferent data path from the default 'path/to/hubmap-organ-segmentation/data'",
        default=CFG.data_path
    )

    parser.add_argument(
        "--num_class",
        type=int,
        help="Number of different classes to segment",
        default=CFG.num_classes
    )

    parser.add_argument(
        "--img_size",
        type=int,
        help="Image size to use",
        default=CFG.img_size
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Specify a diferent model path from the default 'path/to/hubmap-organ-segmentation/models'",
        default=CFG.model_path
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for random generation",
        default=CFG.seed
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Relative size (0-1) of the test set",
        default=CFG.test_size
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        help="Number of folds for cross validation",
        default=CFG.n_folds
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model to train",
        default=CFG.model
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
        default=CFG.epochs
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to train",
        default=CFG.batch_size
    )

    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to train",
        default=CFG.loss
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to train",
        default=CFG.optimizer
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Initial learning rate",
        default=CFG.lr
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay for the learning rate",
        default=CFG.weight_decay
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler for the learning rate",
        default=CFG.scheduler
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device (CUDA or cpu)",
        default=CFG.device
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of threads use in loading data",
        default=CFG.num_workers
    )

    args = parser.parse_args()
    for key,value in vars(args).iteritems():
        setattr(CFG, key, value)

    create_datasets()
    model_pipeline()