import wandb
from main import CFG
from models.train_model import train, test

def model_pipeline():
    wandb.login()
    config = dict(
        model = CFG.model,
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        loss=CFG.loss,
        optimizer=CFG.optimizer,
        learning_rate=CFG.lr,
        scheduler=CFG.scheduler
    )
    # tell wandb to get started
    with wandb.init(project="hubmap-organ-segmentation", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        model = train()
        test_loss, test_iou = test(model)
        wandb.log(
            {"test_loss": test_loss,
            "test_iou": test_iou}
        )
