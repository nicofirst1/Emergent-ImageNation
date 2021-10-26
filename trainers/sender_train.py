import torch
from egg import core
from egg.core import LoggingStrategy, CheckpointSaver, ProgressBarLogger
from torch.optim import Adam
from torch.utils.data import DataLoader

from Parameters import SenderTrainParams, PathParams
from arhcs.sender import get_sender, get_dalle_params
from dataset import CaptionDataset
from utils import CustomWandbLogger


class SenderTrain(torch.nn.Module):
    """
    Sender train logic for egg
    Simply gets the data (images, text, mask), gets it to dalle and return loss and interactions
    """

    def __init__(self, dalle,
                 train_logging_strategy: LoggingStrategy = None,
                 test_logging_strategy: LoggingStrategy = None, ):
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

    def forward(self, images, text, mask, something):
        loss = self.dalle(text, images, mask=mask, return_loss=True)

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
        )

        return loss, interaction


if __name__ == '__main__':
    # get configurations
    core.init()
    st_params = SenderTrainParams()
    pt_params = PathParams()
    model_config = get_dalle_params()

    # get dataloader
    train_data = CaptionDataset(pt_params.preprocessed_dir, "TRAIN")
    val_data = CaptionDataset(pt_params.preprocessed_dir, "VAL")

    train_dl = DataLoader(train_data, batch_size=st_params.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_data, batch_size=st_params.batch_size, shuffle=True, drop_last=True)

    # initialize dalle and game
    dalle = get_sender(model_config)
    sender_train = SenderTrain(dalle)

    # optimizer
    opt = Adam(dalle.parameters(), lr=st_params.lr)

    # init callbacks
    wandb_logget = CustomWandbLogger(log_step=100, image_log_step=1000, dalle=dalle,
                                     project='dalle_train_transformer', config=model_config,
                                     dir=pt_params.wandb_dir, opts={})

    checkpoint_logger = CheckpointSaver(checkpoint_path=st_params.checkpoint, max_checkpoints=3)

    progressbar = ProgressBarLogger(n_epochs=st_params.epochs, train_data_len=len(train_data),
                                    test_data_len=len(val_data), use_info_table=False)

    callbacks = [
        wandb_logget,
        checkpoint_logger,
        progressbar
    ]

    # training

    trainer = core.Trainer(
        game=sender_train,
        optimizer=opt,
        train_data=train_dl,
        validation_data=val_dl,
        device=st_params.cuda,
        grad_norm=True,
        callbacks=callbacks

    )

    trainer.train(st_params.epochs)
