import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import wandb
from dalle_pytorch.tokenizer import tokenizer
from nltk.translate.bleu_score import corpus_bleu
from rich.progress import track
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from Parameters import PathParams, ReceiverTrainParams, DataParams
from arhcs.receiver import DecoderWithAttention, Encoder
# Data parameters
from dataset import CaptionDataset


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


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


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    # number of steps (* log_step) to log images and captions
    log_image_step = 1

    # Batches
    for i, (imgs, caps, caplens) in track(enumerate(train_loader), total=len(train_loader),
                                          description=f"Training epoch {epoch}..."):

        # Move to GPU, if available
        imgs = imgs.to(rt_params.device)
        caps = caps.to(rt_params.device)
        caplens = caplens.to(rt_params.device)

        # for logging
        img= imgs[0]

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        scores_copy = scores

        # move everything to same device
        targets = targets.to(rt_params.device)
        scores = scores.to(rt_params.device)
        alphas = alphas.to(rt_params.device)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += rt_params.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if rt_params.grad_clip is not None:
            clip_gradient(decoder_optimizer, rt_params.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, rt_params.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Print status
        if i % rt_params.print_freq == 0:
            top5 = accuracy(scores, targets, 5)

            log = {
                'epoch': epoch,
                'iter': i,
                'train_loss': loss.item(),
                'train_top5': top5
            }

            if log_image_step == 0:

                tg = caps[0]
                tg = tokenizer.decode(tg)

                _, preds = torch.max(scores_copy[0], dim=1)
                preds = preds.tolist()
                preds = tokenizer.decode(preds)

                #caption = f"Original : {tg}\nPredicted: {preds}"

                log = {
                    **log,
                    'image': wandb.Image(img, caption=preds)
                }

                log_image_step = 10

            else:
                log_image_step -= 1

            if not rt_params.debug:
                wandb.log(log)



def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    log_image_step = 10

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in track(enumerate(val_loader), total=len(val_loader),
                                                       description=f"Validating epoch {epoch}..."):

            # Move to device, if available
            imgs = imgs.to(rt_params.device)
            caps = caps.to(rt_params.device)
            caplens = caplens.to(rt_params.device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()

            # move everything to same device
            targets = targets.to(rt_params.device)
            scores = scores.to(rt_params.device)
            alphas = alphas.to(rt_params.device)

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += rt_params.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            if i % rt_params.print_freq == 0:
                top5 = accuracy(scores, targets, 5)

                log = {
                    'epoch': epoch,
                    'iter': i,
                    'val_loss': loss.item(),
                    'val_top5': top5
                }

                if log_image_step == 0:

                    img = imgs[0]
                    tg = caps[0]
                    tg = tokenizer.decode(tg)

                    _, preds = torch.max(scores_copy[0], dim=1)
                    preds = preds.tolist()
                    preds = tokenizer.decode(preds)

                    #caption = f"Original : {tg}\nPredicted: {preds}"

                    log = {
                        **log,
                        'image': wandb.Image(img, caption=preds)
                    }

                    log_image_step = 10

                else:
                    log_image_step -= 1

                if not rt_params.debug:
                    wandb.log(log)


            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: tokenizer.decode(c), img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)


        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        log = {
            **log,
            'bleu': bleu4
        }
        if not rt_params.debug:
            wandb.log(log)

    return bleu4


if __name__ == '__main__':
    """
        Training and validation.
        """

    rt_params = ReceiverTrainParams()
    data_params = DataParams()
    pt_params = PathParams()

    # Initialize / load checkpoint
    if rt_params.checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=rt_params.attention_dim,
                                       embed_dim=rt_params.emb_dim,
                                       decoder_dim=rt_params.decoder_dim,
                                       vocab_size=data_params.vocab_size,
                                       dropout=rt_params.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=rt_params.decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(rt_params.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=rt_params.encoder_lr) if rt_params.fine_tune_encoder else None

    else:
        checkpoint = torch.load(rt_params.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if rt_params.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(rt_params.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=rt_params.encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(rt_params.device)
    encoder = encoder.to(rt_params.device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(rt_params.device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    data_params = PathParams()

    train_data = CaptionDataset(data_params.preprocessed_dir, "TRAIN", transform=transform)
    val_data = CaptionDataset(data_params.preprocessed_dir, "VAL", transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=rt_params.batch_size, shuffle=True, num_workers=rt_params.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=rt_params.batch_size, shuffle=True, num_workers=rt_params.workers, pin_memory=False)

    epochs_since_improvement = 0
    best_bleu4 = 0.  # BLEU-4 score right now

    if not rt_params.debug: run = wandb.init(project='receiver_train', config=rt_params.__dict__,
                                             dir=pt_params.wandb_dir)

    # Epochs
    for epoch in range(rt_params.start_epoch, rt_params.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if rt_params.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(rt_params.data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
