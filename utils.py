import os

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
        self.dalle = dalle
        self.model_config = config

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

        if batch_id % self.image_log_step == 0:
            sample_text = logs.labels[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            mask= logs.message_length[:1]

            sample_text = sample_text.to(self.trainer.device)
            mask = mask.to(self.trainer.device)

            image = self.dalle.generate_images(
                sample_text,
                mask=mask,
                filter_thres=0.9  # topk sampling at 0.9
            )
            image = image[0]
            wandb_log = {
                **wandb_log,
                f'{flag}_image': wandb.Image(image, caption=decoded_text)
            }

        self.log_to_wandb(wandb_log, commit=True)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(self.model_config))
        model_artifact.add_file('dalle.pt')
