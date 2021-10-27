import os

import torch
import wandb
from dalle_pytorch.tokenizer import tokenizer
from egg.core import Interaction
from egg.core.callbacks import WandbLogger


class CustomWandbLogger(WandbLogger):

    def __init__(self, log_step, image_log_step, dalle, dir, config, **kwargs):

        # create wandb dir if not existing
        if not os.path.isdir(dir):
            os.mkdir(dir)

        super(CustomWandbLogger, self).__init__(dir=dir, config=config, **kwargs)

        self.log_step = log_step
        self.image_log_step = image_log_step
        self.sender = dalle
        self.model_config = config

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
            sample_text,
            mask=mask,
            filter_thres=0.9  # topk sampling at 0.9
        )
        image = image[0]
        wandb_log = {
            f'{flag}_sender': wandb.Image(image, caption=decoded_text)
        }
        return wandb_log

    def receiver_image_log(self, flag, logs):
        """
        Use the reciever to generate a caption from the image
        :param flag:
        :param logs:
        :return:
        """

        preds = tokenizer.decode(logs.receiver_output)
        img = logs.sender_input

        return {f'{flag}_receiver': wandb.Image(img, caption=preds)}

    def on_batch_end(
            self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if batch_id % self.log_step != 0:
            return

        flag = "training" if is_training else "validation"

        wandb_log = {
            f"{flag}_loss": loss,
            f"{flag}_iter": batch_id,
        }

        if self.sender is None:
            top5 = accuracy(logs.aux['scores'], logs.aux['targets'], 5)
            wandb_log[f"{flag}_top5"] = top5

        # image logging
        if batch_id % self.image_log_step == 0:
            if self.sender is not None:
                img_log = self.sender_image_log(flag, logs)
            else:
                img_log = self.receiver_image_log(flag, logs)

            wandb_log.update(img_log)

        self.log_to_wandb(wandb_log, commit=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(self.model_config))
        model_artifact.add_file('dalle.pt')

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        a=1
        #todo: add    bleu4 = corpus_bleu(references, hypotheses)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


from sentence_transformers import SentenceTransformer, util


def SBERT_loss():
    def inner(true_description, receiver_output):
        """
        Estimate the Cosine similarity among sentences
        using SBERT
        https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        https://arxiv.org/abs/1908.10084
        https://github.com/UKPLab/sentence-transformers
        """

        emb1 = model.encode(receiver_output)
        emb2 = model.encode(true_description)

        loss = -util.cos_sim(emb1, emb2)
        print(loss)

        return loss

    model = SentenceTransformer('all-MiniLM-L6-v2')
    return inner