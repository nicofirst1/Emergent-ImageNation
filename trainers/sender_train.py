import os

import torch
import wandb
from dalle_pytorch.tokenizer import tokenizer
from rich.progress import track
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from Parameters import SenderTrainParams, PathParams
from arhcs.sender import get_sender, get_dalle_params
from dataset import CaptionDataset

st_params = SenderTrainParams()
pt_params=PathParams()

model_config = get_dalle_params()
weights = "./dalle.pt"


train_data = CaptionDataset(pt_params.preprocessed_dir, "TRAIN")

dl = DataLoader(train_data, batch_size=st_params.batch_size, shuffle=True, drop_last=True)

dalle = get_sender(cuda=st_params.cuda)

if weights is not None:
    weights=torch.load(weights)
    dalle.load_state_dict(weights)
    print("Dalle weights loaded!")

# optimizer

opt = Adam(dalle.parameters(), lr=st_params.lr)

# experiment tracker

if not os.path.isdir(pt_params.wandb_dir):
    os.mkdir(pt_params.wandb_dir)

if not st_params.debug: run = wandb.init(project='dalle_train_transformer', config=model_config, dir=pt_params.wandb_dir)
# training

for epoch in range(st_params.epochs):
    for i, (images, text, mask) in track(enumerate(dl), total=len(dl), description="Batches..."):

        if st_params.cuda:
            text, images, mask = map(lambda t: t.cuda(), (text, images, mask))

        loss = dalle(text, images, mask=mask, return_loss=True)

        loss.backward()
        clip_grad_norm_(dalle.parameters(), st_params.grad_clip_norm)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 100 == 0:
            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 1000 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            image = dalle.generate_images(
                text[:1],
                mask=mask[:1],
                filter_thres=0.9  # topk sampling at 0.9
            )
            image = image[0]
            torch.save(dalle.state_dict(), f'./dalle.pt')
            if not st_params.debug: wandb.save(f'./dalle.pt')

            log = {
                **log,
                'image': wandb.Image(image, caption=decoded_text)
            }

        if not st_params.debug: wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end

    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file('dalle.pt')

torch.save(dalle.state_dict(), f'./dalle-final.pt')
if not st_params.debug: wandb.save('./dalle-final.pt')
model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
model_artifact.add_file('dalle-final.pt')

if not st_params.debug: wandb.finish()
