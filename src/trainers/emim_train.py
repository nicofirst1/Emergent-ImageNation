import torch.optim
import torch.utils.data
from egg import core
from egg.core import LoggingStrategy, ProgressBarLogger, CheckpointSaver
from torch.nn.utils.rnn import pack_padded_sequence
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.transforms import transforms

from src.Parameters import PathParams, ReceiverParams, DataParams, SenderParams, DebugParams
from src.arhcs.receiver import get_recevier
# Data parameters
from src.arhcs.sender import get_sender, get_sender_params
from src.dataset import get_dataloaders
from src.utils import CustomWandbLogger, SBERT_loss


class EmImTrain(torch.nn.Module):
    """
    Sender train logic for egg
    Simply gets the data (images, text, mask), gets it to dalle and return loss and interactions
    """

    def __init__(self, encoder, decoder, sender,
                 train_logging_strategy: LoggingStrategy = None,
                 test_logging_strategy: LoggingStrategy = None, ):
        super(EmImTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sender = sender

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

        self.loss_function = SBERT_loss()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, images, text, mask, something):

        # get image from sender
        sender_images = self.sender.generate_images_trainmode(text, mask=mask)

        # save for log
        sender_img = sender_images[0]
        # normalize and sub std
        sender_images = self.transform(sender_images)

        # call receiver with generated image
        imgs = encoder(sender_images)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, text, mask)

        # for logging
        img = images[0]
        targets = caps_sorted[:, 1:]
        _, preds = torch.max(scores, dim=2)

        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # estimate loss
        loss = self.loss_function(text, preds)


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
            receiver_output=preds[0],
            aux=dict(
                scores=scores,
                sender_img=sender_img,
                targets=targets,
            ),
        )

        torch.cuda.empty_cache()

        return loss, interaction


if __name__ == '__main__':
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

    #################
    #   SENDER
    #################
    # initialize Sender
    model_config = get_sender_params()
    sender = get_sender(model_config)

    # optimizer
    sender_opt = dict(params=filter(lambda p: p.requires_grad, sender.parameters()),
                      lr=st_params.lr)

    #################
    #   RECEIVER
    #################
    # get architecture
    decoder, encoder = get_recevier()

    # initialize optimizers

    dec_opt = dict(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                   lr=rt_params.decoder_lr)

    enc_opt = dict(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                   lr=rt_params.encoder_lr)

    #################
    #   OPTIMIZERS
    #################

    opt_list = [sender_opt, dec_opt]
    if rt_params.fine_tune_encoder:
        opt_list.append(enc_opt)

    joint_optim = torch.optim.Adam(opt_list)

    # Custom dataloaders

    train_dl, val_dl = get_dataloaders()

    #################
    #   CALLBACKS
    #################

    checkpoint_logger = CheckpointSaver(checkpoint_path=rt_params.checkpoint, max_checkpoints=3)

    progressbar = ProgressBarLogger(n_epochs=rt_params.epochs, train_data_len=len(train_dl),
                                    test_data_len=len(val_dl), use_info_table=False)

    callbacks = [
        checkpoint_logger,
        progressbar
    ]

    if not deb_params.debug and True:
        wandb_logger = CustomWandbLogger(log_step=100, image_log_step=1000, dalle=sender,
                                         project='emim_train', model_config={},
                                         dir=pt_params.wandb_dir, opts={}, log_type="emim")
        callbacks.append(wandb_logger)

    # training

    setting = EmImTrain(encoder, decoder, sender)

    trainer = core.Trainer(
        game=setting,
        optimizer=joint_optim,
        train_data=train_dl,
        validation_data=val_dl,
        device=deb_params.device,
        grad_norm=rt_params.grad_clip,
        callbacks=callbacks

    )

    trainer.train(rt_params.epochs)
