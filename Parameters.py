import os

import torch
from torch.backends import cudnn


class ReceiverTrainParams:
    debug = True
    data_name = "receiver"

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not debug else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True if not debug else False  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Training parameters
    start_epoch = 0
    epochs = 120  # number of epochs to train for (if early stopping is not triggered)
    batch_size = 32 if not debug else 2
    workers = 1  # for data-loading; right now, only 1 works with h5py
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    grad_clip = 5.  # clip gradients at an absolute value of
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
    print_freq = 100  # print training/validation stats every __ batches
    fine_tune_encoder = False  # fine-tune encoder?
    checkpoint = None  # path to checkpoint, None if none


class SenderTrainParams:
    debug = False

    ### TRAINING
    EPOCHS = 20
    BATCH_SIZE = 4
    cuda = True

    ### DALLe training params

    LEARNING_RATE = 1e-3
    LR_DECAY_RATE = 0.98
    GRAD_CLIP_NORM = 0.5

    STARTING_TEMP = 1.
    TEMP_MIN = 0.5
    ANNEAL_RATE = 1e-6

    ### DALLE model params
    MODEL_DIM = 128
    TEXT_SEQ_LEN = 256
    DEPTH = 2
    HEADS = 4
    DIM_HEAD = 64
    REVERSIBLE = True


class PathParams:
    base_path = "/home/dizzi/Desktop/coco/"

    preprocessed_dir = os.path.join(base_path, "preprocessed")


class DataParams:
    ### DATA CREATION
    vocab_size = 49408  # used also in dalle setupp
    captions_per_image = 5  # number of captions to keep per image
    min_word_freq = 5
