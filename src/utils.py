import json
import os
from itertools import chain

import wandb
from dalle_pytorch.tokenizer import tokenizer
from egg.core import Interaction
from egg.core.callbacks import WandbLogger
from nltk.translate.bleu_score import sentence_bleu

from src.Parameters import PathParams


class CustomWandbLogger(WandbLogger):
    def __init__(
        self,
        train_log_step,
        val_log_step,
        tokenizer,
        dalle,
        dir,
        model_config,
        log_type,
        **kwargs,
    ):

        # create wandb dir if not existing
        if not os.path.isdir(dir):
            os.mkdir(dir)

        super(CustomWandbLogger, self).__init__(dir=dir, config=model_config, **kwargs)

        self.train_log_step = train_log_step if train_log_step > 0 else 2
        self.val_log_step = val_log_step if val_log_step > 0 else 2
        self.sender = dalle
        self.model_config = model_config
        self.receiver_decoder = None
        self.tokenizer = tokenizer

        assert log_type in ["sender", "receiver", "emim"]
        self.log_type = log_type

        self.epoch = 0

    def sender_image_log(self, flag, logs):
        """
        Use the sender to generate an image from a caption
        :param flag:
        :param logs:
        :return:
        """
        sample_text = logs.labels[:1]
        token_list = sample_text.masked_select(sample_text != 0).tolist()
        decoded_text = tokenizer.decode(token_list)

        mask = logs.message_length[:1]

        sample_text = sample_text.to(self.trainer.device)
        mask = mask.to(self.trainer.device)

        image = self.sender.generate_images(
            sample_text, mask=mask, filter_thres=0.9  # topk sampling at 0.9
        )
        image = image[0]
        wandb_log = {f"{flag}_sender": wandb.Image(image, caption=decoded_text)}
        return wandb_log

    def receiver_image_log(self, flag, logs):
        """
        Use the reciever to generate a caption from the image
        :param flag:
        :param logs:
        :return:
        """

        preds = self.tokenizer.decode(logs.receiver_output)
        img = logs.sender_input

        return {f"{flag}_receiver": wandb.Image(img, caption=preds)}

    def emim_image_log(self, flag, logs):
        """
        Logs both the original pair (image/caption) and the predicted one (sender image and receiver caption)
        :param flag:
        :param logs:
        :return:
        """
        original_caption = logs.labels[0]
        original_image = logs.sender_input

        pred = self.tokenizer.decode(logs.receiver_output[0])

        columns = ["Original Caption", "Predicted caption"]
        table = wandb.Table(columns=columns)
        table.add_data(original_caption, pred)

        return {
            f"{flag}_original": wandb.Image(original_image, caption=original_caption),
            f"{flag}_captions": table,
        }

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):

        flag = "training" if is_training else "validation"

        log_step = self.train_log_step

        if not is_training:
            log_step = self.val_log_step

        image_log_step = log_step * 10

        if logs.is_empty():
            return

        wandb_log = {
            f"{flag}_loss": loss,
            f"{flag}_iter": batch_id * (self.epoch + 1),
            f"{flag}_epoch": self.epoch,
        }

        if self.log_type != "sender":
            top5 = accuracy(logs.aux["scores"], logs.aux["targets"], 5)
            wandb_log[f"{flag}_top5"] = top5

        # image logging
        if batch_id % image_log_step == 0:
            if self.log_type == "sender":
                img_log = self.sender_image_log(flag, logs)
            elif self.log_type == "receiver":
                img_log = self.receiver_image_log(flag, logs)
            else:
                img_log = self.emim_image_log(flag, logs)

            wandb_log.update(img_log)

        self.log_to_wandb(wandb_log, commit=True, step=batch_id)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):

        self.log_bleu_score(logs, "training")
        self.epoch += 1

        model_artifact = wandb.Artifact(
            "trained-game", type="model", metadata=dict(self.model_config)
        )
        model_path = os.path.join(PathParams.checkpoint_emim, f"{self.epoch}.tar")

        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

    def log_bleu_score(self, logs, flag):
        references = logs.aux_input["all_captions"]
        references = [list(zip(*elem)) for elem in references]
        references = list(chain.from_iterable(references))

        hypotesys = logs.receiver_output

        hypotesys = [elem[elem.nonzero()].squeeze().tolist() for elem in hypotesys]
        hypotesys = [self.tokenizer.decode(elem) for elem in hypotesys]

        bleu = 0
        for i in range(len(hypotesys)):
            bleu += sentence_bleu(references[i], hypotesys[i])

        bleu /= i + 1

        self.log_to_wandb(
            {
                f"{flag}_bleu": bleu,
            },
            commit=True,
        )

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):

        self.log_bleu_score(logs, "validation")


def build_translation_vocabulary():
    with open(PathParams.receiver_wordmap_path, "r") as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}

    return word_map, rev_word_map


def dictionary_decode(dictionary):
    def decode(sentence):
        # remove last word
        translated_caption = [dictionary.get(int(elem), "unk") for elem in sentence]

        return " ".join(translated_caption)

    return decode


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    scores = scores.data
    targets = targets.data

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


from sentence_transformers import SentenceTransformer, util


def get_loggings(train_len, val_len, perc=0.01):
    train_step = int(train_len * perc)
    val_step = int(val_len * perc)

    return train_step, val_step


def SBERT_loss(device, output_decoder=tokenizer):
    def inner(true_description, receiver_output):
        """
        Estimate the Cosine similarity among sentences
        using SBERT
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        https://arxiv.org/abs/1908.10084
        https://github.com/UKPLab/sentence-transformers
        """

        def encode(sentences):
            features = model.tokenize(sentences)
            # features = batch_to_device(features, "cuda")
            out_features = model.forward(features)

            return out_features

        if isinstance(output_decoder, dict):
            receiver_output = [
                [output_decoder[int(elem)] for elem in x] for x in receiver_output
            ]
            receiver_output = [" ".join(x) for x in receiver_output]

        else:
            receiver_output = [output_decoder.decode(elem) for elem in receiver_output]

        emb1 = encode(receiver_output)
        emb2 = encode(true_description)

        emb1 = emb1["sentence_embedding"]
        emb2 = emb2["sentence_embedding"]

        loss = -util.cos_sim(emb1, emb2)
        loss = loss.mean().to(device)

        return loss

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return inner


if __name__ == "__main__":
    build_translation_vocabulary()
