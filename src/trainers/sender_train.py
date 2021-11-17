import torch
from egg import core
from egg.core import CheckpointSaver, LoggingStrategy, ProgressBarLogger
from src.archs.sender import get_sender, get_sender_params
from src.dataset import get_dataloaders
from src.Parameters import DebugParams, PathParams, SenderParams
from src.utils import  CustomWandbLogger, get_loggings
from torch.optim import AdamW, lr_scheduler


class SenderTrain(torch.nn.Module):
    """
    Sender train logic for egg
    Simply gets the data (images, text, mask), gets it to dalle and return loss and interactions
    """

    def __init__(
        self,
        dalle,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        super(SenderTrain, self).__init__()
        self.dalle = dalle

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, images, text, mask, something, batch_id):
        loss = self.dalle(text, image=images, mask=mask, return_loss=True)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=None,  # image
            labels=text,
            message_length=mask,
            receiver_input=None,
            aux_input=None,
            message=None,
            receiver_output=None,
            aux={},
            batch_id=batch_id,
        )

        return loss, interaction


if __name__ == "__main__":
    # get configurations
    core.init(params=[])
    st_params = SenderParams()
    pt_params = PathParams()
    deb_params = DebugParams()
    model_config = get_sender_params()

    # get dataloader
    train_dl, val_dl = get_dataloaders()

    # initialize dalle and game
    dalle = get_sender(model_config)
    sender_train = SenderTrain(dalle)

    # optimizer
    opt = AdamW(dalle.parameters(), lr=st_params.lr)

    optimizer_scheduler = lr_scheduler.OneCycleLR(
        opt, max_lr=5e-3, epochs=st_params.epochs, steps_per_epoch=len(train_dl)
    )

    # init callbacks

    checkpoint_logger = CheckpointSaver(
        checkpoint_path=st_params.checkpoint, max_checkpoints=3
    )

    progressbar = ProgressBarLogger(
        n_epochs=st_params.epochs,
        train_data_len=len(train_dl),
        test_data_len=len(val_dl),
        use_info_table=False,
    )

    callbacks = [checkpoint_logger, progressbar]
    train_step, val_step = get_loggings(len(train_dl), len(val_dl), perc=0.01)

    if not deb_params.debug:
        wandb_logger = CustomWandbLogger(
            train_log_step=train_step,
            val_log_step=val_step,
            dalle=dalle,
            project="sender_train",
            model_config=model_config,
            dir=pt_params.wandb_dir,
            opts={},
            log_type="sender",
        )
        callbacks.append(wandb_logger)

    sender_train = SenderTrain(
        dalle,
    )

    # training

    trainer = core.Trainer(
        game=sender_train,
        optimizer=opt,
        train_data=train_dl,
        validation_data=val_dl,
        device=deb_params.device,
        grad_norm=True,
        callbacks=callbacks,
        optimizer_scheduler=optimizer_scheduler,
    )

    trainer.train(st_params.epochs)
