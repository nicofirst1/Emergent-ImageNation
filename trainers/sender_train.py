import os

import torch
from dalle_pytorch.tokenizer import tokenizer
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
from Parameters import  SenderTrainParams
from arhcs.sender import get_sender, get_dalle_params
from dataset import CaptionDataset


params= SenderTrainParams()
model_config=get_dalle_params()
weights=None


base_path = "/home/dizzi/Desktop/coco/"
output_dir = os.path.join(base_path, "preprocessed")

train_data = CaptionDataset(output_dir, "TRAIN")

dl = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, drop_last=True)

dalle = get_sender(cuda=params.cuda)

if weights is not None:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(dalle.parameters(), lr=params.LEARNING_RATE)

# experiment tracker

run = wandb.init(project='dalle_train_transformer', config=model_config)
# training

for epoch in range(params.EPOCHS):
    for i, (images, text, mask) in enumerate(dl):

        if params.cuda:
            text, images, mask = map(lambda t: t.cuda(), (text, images, mask))

        loss = dalle(text, images, mask=mask, return_loss=True)

        loss.backward()
        clip_grad_norm_(dalle.parameters(), params.GRAD_CLIP_NORM)

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

            torch.save(dalle.state_dict(), f'./dalle.pt')
            wandb.save(f'./dalle.pt')

            log = {
                **log,
                'image': wandb.Image(image, caption=decoded_text)
            }

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end

    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file('dalle.pt')

torch.save(dalle.state_dict(), f'./dalle-final.pt')
wandb.save('./dalle-final.pt')
model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
model_artifact.add_file('dalle-final.pt')

wandb.finish()
