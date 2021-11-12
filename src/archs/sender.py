import torch
from dalle_pytorch import DALLE, VQGanVAE
from dalle_pytorch.tokenizer import tokenizer

from src.Parameters import DataParams, SenderParams, DebugParams


class CustomDalle(DALLE):

    def __init__(self, device, use_image, **kwargs):
        super(CustomDalle, self).__init__(**kwargs)
        self.device = device
        self.use_image = use_image

    def forward(
            self,
            text,
            image=None,
            mask=None,
            return_loss=False,
            return_tokens=False,
    ):
        # tokenize captions
        captions = ["<|startoftext|> " + cap + " <|endoftext|>" for cap in text]
        captions = tokenizer.tokenize(captions, context_length=self.text_seq_len)
        captions = torch.LongTensor(captions).to(self.device)

        if not self.use_image:
            image = None

        return super(CustomDalle, self).forward(captions, image=image,
                                                mask=mask,
                                                return_loss=return_loss,
                                                return_tokens=return_tokens)


def get_sender_params():
    params = SenderParams()
    dt = DataParams()
    db = DebugParams()

    return dict(
        num_text_tokens=dt.vocab_size_in,
        text_seq_len=dt.max_text_seq_len,
        dim=params.model_dim,
        depth=params.depth,
        heads=params.heads,
        dim_head=params.dim_head,
        reversible=params.reversible,
        device=db.device,
        use_image=params.use_image,
    )


def get_sender(dalle_params=None) -> DALLE:
    if dalle_params is None:
        dalle_params = get_sender_params()

    vae = VQGanVAE()  # loads pretrained taming Transformer

    dalle = CustomDalle(vae=vae, **dalle_params)

    return dalle
