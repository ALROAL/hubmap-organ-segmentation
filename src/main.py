import argparse
from data.create_datasets import create_datasets
from models.pipeline import model_pipeline
import json
from PATHS import CONFIG_JSON_PATH
from config import CFG


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates the data and train a model on HuBMAP + HPA dataset"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Specify a diferent data path from the default 'path/to/hubmap-organ-segmentation/data'",
        default=CFG["data_path"]
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of different classes to segment",
        default=CFG["num_classes"]
    )

    parser.add_argument(
        "--img_size",
        type=int,
        help="Image size to use",
        default=CFG["img_size"]
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Specify a diferent model path from the default 'path/to/hubmap-organ-segmentation/models'",
        default=CFG["model_path"]
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for random generation",
        default=CFG["seed"]
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Relative size (0-1) of the test set",
        default=CFG["test_size"]
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        help="Number of folds for cross validation",
        default=CFG["n_folds"]
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model to train",
        default=CFG["model"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
        default=CFG["epochs"]
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to train",
        default=CFG["batch_size"]
    )

    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to train",
        default=CFG["loss"]
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to train",
        default=CFG["optimizer"]
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Initial learning rate",
        default=CFG["lr"]
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay for the learning rate",
        default=CFG["weight_decay"]
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler for the learning rate",
        default=CFG["scheduler"]
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device (CUDA or cpu)",
        default=CFG["device"]
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of threads use in loading data",
        default=CFG["num_workers"]
    )

    args = parser.parse_args()
    for key,value in vars(args).items():
        CFG[key] = value

    with open(CONFIG_JSON_PATH, 'w') as f:
        json.dump(CFG, f, indent=1)

    # create_datasets()
    # model_pipeline()