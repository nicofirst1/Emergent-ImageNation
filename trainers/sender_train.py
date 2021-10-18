import os

import wandb
from dalle_pytorch.tokenizer import tokenizer
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from arhcs.sender import dalle, DEPTH, HEADS, DIM_HEAD
from dataset import CaptionDataset

IMAGE_SIZE = 128
IMAGE_PATH = './'

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.98
GRAD_CLIP_NORM = 0.5

STARTING_TEMP = 1.
TEMP_MIN = 0.5
ANNEAL_RATE = 1e-6

NUM_IMAGES_SAVE = 4
RESUME=False
weights=None


base_path = "/home/dizzi/Desktop/coco/"
output_dir = os.path.join(base_path, "preprocessed")

train_data = CaptionDataset(output_dir, "", "TRAIN")

dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


if RESUME:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(dalle.parameters(), lr=LEARNING_RATE)

# experiment tracker


model_config = dict(
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD
)

run = wandb.init(project='dalle_train_transformer', resume=RESUME, config=model_config)

# training

for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t.cuda(), (text, images, mask))

        loss = dalle(text, images, mask=mask, return_loss=True)

        loss.backward()
        clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            print(epoch, i, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 100 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            image = dalle.generate_images(
                text[:1],
                mask=mask[:1],
                filter_thres=0.9  # topk sampling at 0.9
            )

            save_model(f'./dalle.pt')
            wandb.save(f'./dalle.pt')

            log = {
                **log,
                'image': wandb.Image(image, caption=decoded_text)
            }

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end

    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file('dalle.pt')
    run.log_artifact(model_artifact)

save_model(f'./dalle-final.pt')
wandb.save('./dalle-final.pt')
model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
model_artifact.add_file('dalle-final.pt')
run.log_artifact(model_artifact)

wandb.finish()
