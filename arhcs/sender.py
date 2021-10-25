from dalle_pytorch import DALLE, VQGanVAE

from Parameters import DataParams, SenderTrainParams


def get_dalle_params():
    params = SenderTrainParams()
    dt= DataParams()

    return dict(
        num_text_tokens=dt.vocab_size,
        text_seq_len=dt.max_text_seq_len,
        dim=params.model_dim,
        depth=params.depth,
        heads=params.heads,
        dim_head=params.dim_head,
        reversible=params.reversible
    )


def get_sender(dalle_params=None, cuda=False) -> DALLE:
    if dalle_params is None:
        dalle_params = get_dalle_params()

    vae = VQGanVAE()  # loads pretrained taming Transformer

    dalle = DALLE(vae=vae, **dalle_params)

    if cuda:
        dalle = dalle.cuda()

    return dalle
