import torch.optim
import torch.utils.data
from egg import core
from egg.core import CheckpointSaver, LoggingStrategy, ProgressBarLogger
from src.archs.receiver import get_recevier
# Data parameters
from src.archs.sender import get_sender, get_sender_params
from src.dataset import get_dataloaders
from src.Parameters import (DataParams, DebugParams, PathParams,
                            ReceiverParams, SenderParams)
from src.utils import (CustomLogging, CustomWandbLogger, SBERT_loss,
                       get_loggings)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import lr_scheduler
from torchvision.transforms import transforms


class EmImTrain(torch.nn.Module):
    """
    Sender train logic for egg
    Simply gets the data (images, text, mask), gets it to dalle and return loss and interactions
    """

    def __init__(
        self,
        encoder,
        decoder,
        sender,
        device,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        super(EmImTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sender = sender
        self.device = device

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

        self.loss_function = SBERT_loss(self.device)
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, images, text, mask, something, batch_id):
        # get tokens from transformer inside sender
        # [batch size, -1, model dim]
        # remember that [:,:,text_seq_len:] (first text_seq_len on dim 3 ) are relative to text, while others to img
        sender_tokens = self.sender(text, mask=mask, image=images, return_tokens=True)

        # normalize and sub std
        images = self.transform(images)

        # call receiver with generated image
        encoded_img = encoder(images)

        # concat together sender embedding and encoder ones.
        # remember that sender model dim must be equal to resnet output dim for concat
        receiver_input = torch.cat((encoded_img, sender_tokens), dim=1)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            receiver_input, text, mask
        )

        # for logging
        img = images[0]
        targets = caps_sorted[:, 1:]
        _, preds = torch.max(scores, dim=2)

        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(
            targets, decode_lengths, batch_first=True
        )

        # estimate loss
        loss = self.loss_function(text, preds)

        preds_log = preds[0]

        mask, preds_log, scores, targets = map(
            lambda c: c.detach().to("cpu"), [ mask, preds_log, scores, targets]
        )

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
            receiver_output=preds_log,
            aux=dict(
                scores=scores,
                targets=targets,
            ),
            batch_id=batch_id,
        )

        return loss, interaction


if __name__ == "__main__":
    """
    Training and validation.
    """

    # init parameters
    core.init(params=[])
    st_params = SenderParams()
    rt_params = ReceiverParams()
    data_params = DataParams()
    pt_params = PathParams()
    deb_params = DebugParams()

    # Custom dataloaders

    train_dl, val_dl = get_dataloaders()

    #################
    #   SENDER
    #################
    # initialize Sender
    model_config = get_sender_params()
    sender = get_sender(model_config)

    # optimizer
    sender_opt = dict(
        params=filter(lambda p: p.requires_grad, sender.parameters()), lr=st_params.lr
    )

    #################
    #   RECEIVER
    #################
    # get architecture
    decoder, encoder, tokenizer = get_recevier()

    # initialize optimizers

    dec_opt = dict(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=rt_params.decoder_lr,
    )

    enc_opt = dict(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=rt_params.encoder_lr,
    )

    #################
    #   OPTIMIZERS
    #################

    opt_list = [sender_opt, dec_opt]
    if rt_params.fine_tune_encoder:
        opt_list.append(enc_opt)

    joint_optim = torch.optim.AdamW(opt_list)

    optimizer_scheduler = lr_scheduler.OneCycleLR(
        joint_optim, max_lr=5e-3, epochs=rt_params.epochs, steps_per_epoch=len(train_dl)
    )

    #################
    #   CALLBACKS
    #################

    checkpoint_logger = CheckpointSaver(
        checkpoint_path=rt_params.checkpoint, max_checkpoints=3
    )

    progressbar = ProgressBarLogger(
        n_epochs=rt_params.epochs,
        train_data_len=len(train_dl),
        test_data_len=len(val_dl),
        use_info_table=False,
    )

    callbacks = [checkpoint_logger, progressbar]
    train_step, val_step = get_loggings(len(train_dl), len(val_dl), perc=0.01)

    if  True:
        wandb_logger = CustomWandbLogger(
            train_log_step=train_step,
            val_log_step=val_step,
            dalle=sender,
            project="emim_train",
            model_config={},
            dir=pt_params.wandb_dir,
            opts={},
            log_type="emim",
            mode="disabled" if deb_params.debug else "online",
            tokenizer=tokenizer,
        )
        callbacks.append(wandb_logger)

    # training

    setting = EmImTrain(
        encoder,
        decoder,
        sender,
        deb_params.device,
        train_logging_strategy=CustomLogging(train_step),
        test_logging_strategy=CustomLogging(val_step),
    )

    trainer = core.Trainer(
        game=setting,
        optimizer=joint_optim,
        train_data=train_dl,
        validation_data=val_dl,
        device=deb_params.device,
        grad_norm=rt_params.grad_clip,
        callbacks=callbacks,
        optimizer_scheduler=optimizer_scheduler,
    )

    trainer.train(rt_params.epochs)
