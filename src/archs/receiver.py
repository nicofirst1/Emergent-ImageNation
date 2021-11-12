from typing import List

import torch
import torchvision
from torch import nn

from src.Parameters import DataParams, DebugParams, PathParams, ReceiverParams
from src.utils import build_translation_vocabulary


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(
            pretrained=True
        )  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # print(images.shape)
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(
            out
        )  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(
            0, 2, 3, 1
        )  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        encoder_dim = out.size(-1)
        batch_size = out.size(0)

        out = out.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)

        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # linear layer to transform decoder's output
        self.full_att = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
            self,
            attention_dim,
            embed_dim,
            decoder_dim,
            vocab_size_in,
            vocab_size_out,
            device,
            tokenizer,
            encoder_dim=512,
            dropout=0.5,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size_in: size of vocabulary input, the vocab with which the embedding was trained
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size_in = vocab_size_in
        self.vocab_size_out = vocab_size_out
        self.dropout = dropout
        self.device = device

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim
        )  # attention network
        self.tokenizer = tokenizer

        self.embedding = nn.Embedding(vocab_size_in, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True
        )  # decoding LSTMCell
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim, encoder_dim
        )  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim, vocab_size_out
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size_out

        captions = self.tokenizer.encode(captions)

        # Flatten image
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            self.device
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(
            self.device
        )

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(
                self.f_beta(h[:batch_size_t])
            )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, sort_ind


class Tokenizer:
    """
    Tokenize sentences using predefine word mapping
    """

    def __init__(self, w2i_map, i2w_map, encoding_len, device):
        self.w2i_map = w2i_map
        self.i2w_map = i2w_map
        self.encoding_len = encoding_len
        self.device = device

    def encode(self, sentences: List[str]) -> torch.Tensor:
        sentences = [
            [
                self.w2i_map.get(elem, self.w2i_map["<unk>"])
                # get the id for the word if present, if not get id for unk
                for elem in x.split()  # split sentences into list  of words
            ]
            for x in sentences
        ]
        # pad
        sentences = [
            x + [self.w2i_map["<pad>"]] * (self.encoding_len - len(x))
            # pad in order to get to encoding_len
            for x in sentences
        ]
        # move back to device
        sentences = torch.as_tensor(sentences).to(self.device)
        return sentences

    def decode(self, sentence: torch.Tensor):
        sentence = [self.i2w_map.get(int(elem), "unk") for elem in sentence]

        return " ".join(sentence)


def get_recevier():
    rec_params = ReceiverParams()
    data_params = DataParams()
    deb_params = DebugParams()

    encoder = Encoder()
    encoder.fine_tune(rec_params.fine_tune_encoder)

    vocab_size_in = data_params.vocab_size_in
    vocab_size_out = vocab_size_in
    encoder_dim = encoder.resnet[-1][-1].conv3.out_channels

    if rec_params.load_checkpoint:
        # use values the decoder was trained with
        vocab_size_in = 9490
        vocab_size_out = vocab_size_in
        encoder_dim = 2048

    # build internal word tokenizer
    w2i, i2w = build_translation_vocabulary()
    token = Tokenizer(w2i, i2w, data_params.max_text_seq_len, deb_params.device)

    decoder = DecoderWithAttention(
        attention_dim=rec_params.attention_dim,
        embed_dim=rec_params.emb_dim,
        decoder_dim=rec_params.decoder_dim,
        vocab_size_in=vocab_size_in,
        vocab_size_out=vocab_size_out,
        dropout=rec_params.dropout,
        device=deb_params.device,
        encoder_dim=encoder_dim,
        tokenizer=token,
    )

    if rec_params.load_checkpoint:
        checkpoint = torch.load(PathParams.receiver_decoder_model_path)
        decoder.on_load_checkpoint(checkpoint)

    decoder = decoder.to(deb_params.device)
    encoder = encoder.to(deb_params.device)

    return decoder, encoder, token
