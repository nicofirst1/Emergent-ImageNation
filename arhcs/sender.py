from dalle_pytorch import DALLE, VQGanVAE

from Parameters import DataParams, SenderTrainParams


def get_dalle_params():
    params = SenderTrainParams()
    dt= DataParams()

    return dict(
        num_text_tokens=dt.vocab_size,
        text_seq_len=params.TEXT_SEQ_LEN,
        dim=params.MODEL_DIM,
        depth=params.DEPTH,
        heads=params.HEADS,
        dim_head=params.DIM_HEAD,
        reversible=params.REVERSIBLE
    )


def get_sender(dalle_params=None, cuda=False) -> DALLE:
    if dalle_params is None:
        dalle_params = get_dalle_params()

    vae = VQGanVAE()  # loads pretrained taming Transformer

    dalle = DALLE(vae=vae, **dalle_params)

    if cuda:
        dalle = dalle.cuda()

    return dalle
