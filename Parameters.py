

class SenderTrainParams:
    ### TRAINING
    EPOCHS = 20
    BATCH_SIZE = 8
    cuda = True

    ### DALLE params

    LEARNING_RATE = 1e-3
    LR_DECAY_RATE = 0.98
    GRAD_CLIP_NORM = 0.5

    STARTING_TEMP = 1.
    TEMP_MIN = 0.5
    ANNEAL_RATE = 1e-6


class Parameters:
    ### DATA CREATION
    vocab_size = 11000  # used also in dalle setupp
    captions_per_image = 5  # number of captions to keep per image
    min_word_freq = 5
    max_len = 256  # max words length for caption in image

   ### DALLE params
    MODEL_DIM = 257
    TEXT_SEQ_LEN = 256
    DEPTH = 2
    HEADS = 4
    DIM_HEAD = 64
    REVERSIBLE = True
