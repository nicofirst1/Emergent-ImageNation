import argparse
import inspect
import os

import termcolor
import torch
from torch.backends import cudnn


class Params:

    ##########################
    #    METHODS
    ##########################

    def __init__(self):
        # print(f"{self.__class__.__name__} initialized")

        self.__initialize_dirs()

        # change values based on argparse
        self.__parse_args()

    def __parse_args(self):
        """
        Use argparse to change the default values in the param class
        """

        att = self.__get_attributes()

        """Create the parser to capture CLI arguments."""
        parser = argparse.ArgumentParser()

        # for every attribute add an arg instance
        for k, v in att.items():
            parser.add_argument(
                "--" + k.lower(),
                type=type(v),
                default=v,
            )

        args, unk = parser.parse_known_args()
        for k, v in vars(args).items():
            self.__setattr__(k, v)

    def __get_attributes(self):
        """
        Get a dictionary for every attribute that does not have "filter_str" in it
        :return:
        """

        # get every attribute
        attributes = inspect.getmembers(self)
        # filter based on double underscore
        filter_str = "__"
        attributes = [elem for elem in attributes if filter_str not in elem[0]]
        # convert to dict
        attributes = dict(attributes)

        return attributes

    def __initialize_dirs(self):
        """
        Initialize all the directories  listed above
        :return:
        """
        variables = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        for var in variables:
            if var.lower().endswith("dir"):
                path = getattr(self, var)
                if not os.path.exists(path):
                    termcolor.colored(f"Mkdir {path}", "yellow")
                    os.makedirs(path)


class DebugParams(Params):
    """
    Parameters relative to the debugging phase.
    Irrelevant if debug=False
    """

    debug = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 4  # for data-loading; right now, only 1 works with h5py
    batch_size = 16
    pin_memory = True

    def __init__(self):
        super(DebugParams, self).__init__()

        if self.debug:
            self.device = torch.device("cpu")
            cudnn.benchmark = False
            self.workers = 0
            self.batch_size = 2
            self.pin_memory = False
        else:
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)
            torch.backends.cudnn.benchmark = True


class ReceiverParams(Params):
    data_name = "receiver"

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5

    # Training parameters
    epochs = 120  # number of epochs to train for (if early stopping is not triggered)
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    grad_clip = 5.0  # clip gradients at an absolute value of
    alpha_c = 1.0  # regularization parameter for 'doubly stochastic attention', as in the paper
    fine_tune_encoder = False  # fine-tune encoder?
    checkpoint = "./checkpoint_receiver.pth.tar"  # path to checkpoint, None if none

    load_checkpoint = True


class SenderParams(Params):
    ### TRAINING
    epochs = 20

    ### DALLe training params

    lr = 1e-3
    lr_decay = 0.98
    grad_clip_norm = 0.5

    starting_temp = 1.0
    temp_min = 0.5
    anneal_rate = 1e-6

    ### DALLE model params
    model_dim = 2048  # for the emim train be sure to match resnet's output dim
    depth = 2
    heads = 4
    dim_head = 64
    reversible = True

    checkpoint = "./checkpoint_sender.pth.tar"  # path to checkpoint, None if none
    use_image = True


class PathParams(Params):
    working_dir = os.getcwd().split("src")[0]

    preprocessed_dir = os.path.join(working_dir, "preprocessed")

    coco_path = "/home/dizzi/Desktop/coco/"

    wandb_dir = "./wandb_metadata"

    receiver_decoder_model_path = os.path.join(
        preprocessed_dir, "receiver_decoder_check.tar"
    )
    receiver_wordmap_path = os.path.join(
        preprocessed_dir, "WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
    )


class DataParams(Params):
    ### DATA CREATION
    vocab_size_in = 49408  # used also in dalle setup
    captions_per_image = 5  # number of captions to keep per image
    min_word_freq = 5
    max_text_seq_len = 64

    # when true use url in the datset and load them during training, less memory but more time
    generate_data_url = False
