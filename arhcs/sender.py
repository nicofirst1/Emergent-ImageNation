import torch
from dalle_pytorch import DALLE, VQGanVAE

MODEL_DIM = 512
TEXT_SEQ_LEN = 256
DEPTH = 2
HEADS = 4
DIM_HEAD = 64
REVERSIBLE = True

VOCAB_SIZE=999

dalle_params = dict(
    num_text_tokens=VOCAB_SIZE,
    text_seq_len=TEXT_SEQ_LEN,
    dim=MODEL_DIM,
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD,
    reversible=REVERSIBLE
)

vae = VQGanVAE()  # loads pretrained taming Transformer

dalle = DALLE(vae=vae, **dalle_params).cuda()


