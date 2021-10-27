import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from egg import core
from egg.core import LoggingStrategy, ProgressBarLogger, CheckpointSaver
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from Parameters import PathParams, ReceiverParams, DataParams
from arhcs.receiver import get_recevier
# Data parameters
from dataset import get_dataloaders
from utils import CustomWandbLogger


class ReceiverTrain(torch.nn.Module):
    """
    Sender train logic for egg
    Simply gets the data (images, text, mask), gets it to dalle and return loss and interactions
    """

    def __init__(self, encoder, decoder,
                 train_logging_strategy: LoggingStrategy = None,
                 test_logging_strategy: LoggingStrategy = None, ):
        super(ReceiverTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

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

        self.loss_func = nn.CrossEntropyLoss().to(rt_params.device)

    def forward(self, images, text, mask, something):
        # Forward prop.
        imgs = encoder(images)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, text, mask)

        # for logging
        img = images[0]
        _, preds = torch.max(scores[0], dim=1)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # move everything to same device
        targets = targets.to(rt_params.device)
        scores = scores.to(rt_params.device)
        alphas = alphas.to(rt_params.device)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = self.loss_func(scores, targets)

        # Add doubly stochastic attention regularization
        loss += rt_params.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=img,  # image
            labels=text,
            message_length=mask,
            receiver_input=None,
            aux_input=None,
            message=None,
            receiver_output=preds,
            aux=dict(
                scores=scores,
                targets=targets
            ),
        )

        return loss, interaction


if __name__ == '__main__':
    """
        Training and validation.
    """
    core.init(params=[])
    rt_params = ReceiverParams()
    data_params = DataParams()
    pt_params = PathParams()

    # get architecture
    decoder, encoder = get_recevier()
    receiver_train = ReceiverTrain(encoder, decoder)

    # initialize optimizers

    dec_opt = dict(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                   lr=rt_params.decoder_lr)

    enc_opt = dict(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                   lr=rt_params.encoder_lr)

    opt_list = [dec_opt]
    if rt_params.fine_tune_encoder:
        opt_list.append(enc_opt)

    joint_optim = torch.optim.Adam(opt_list)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    train_dl, val_dl = get_dataloaders(transform=transform)

    # init callbacks
    wandb_logger = CustomWandbLogger(log_step=100, image_log_step=1000, dalle=None,
                                     project='receiver_train', config={},
                                     dir=pt_params.wandb_dir, opts={})

    checkpoint_logger = CheckpointSaver(checkpoint_path=rt_params.checkpoint, max_checkpoints=3)

    progressbar = ProgressBarLogger(n_epochs=rt_params.epochs, train_data_len=len(train_dl),
                                    test_data_len=len(val_dl), use_info_table=False)

    callbacks = [
        wandb_logger,
        checkpoint_logger,
        progressbar
    ]

    # training

    trainer = core.Trainer(
        game=receiver_train,
        optimizer=joint_optim,
        train_data=train_dl,
        validation_data=val_dl,
        device=rt_params.device,
        grad_norm=rt_params.grad_clip,
        callbacks=callbacks

    )

    trainer.train(rt_params.epochs)
