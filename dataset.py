import json
import os
import pickle
import random
from random import seed, choice, sample

import cv2
import h5py
import numpy as np
import torch
from dalle_pytorch.tokenizer import SimpleTokenizer
from imageio import imread
from rich.progress import track
from torch.utils.data import Dataset

from Parameters import DataParams


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None, seed=42):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
        files = [elem for elem in files if self.split in elem]
        assert len(files) == 3, f"Found {len(files)} files in {data_folder}, need 3 (IMAGES, CAPTIONS, CAPLEN)."

        hdf5 = [elem for elem in files if "IMAGES" in elem][0]
        captions = [elem for elem in files if "CAPTIONS" in elem][0]
        caplens = [elem for elem in files if "CAPLENS" in elem][0]

        self.h = h5py.File(os.path.join(data_folder, hdf5), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, captions), 'rb') as file:
            self.captions = pickle.load(file)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, caplens), 'r') as file:
            self.caplens = json.load(file)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        random.seed(seed)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        cap_index = random.randint(0, len(self.captions[i]) - 1)
        caption = torch.LongTensor(self.captions[i][cap_index])
        caplen = torch.zeros(caption.size())
        caplen[:self.caplens[i][cap_index]] = 1

        if False:  # set to true if you want to debug the image/caption pair
            from torchvision.transforms import ToPILImage

            # show the image
            ToPILImage()(img).show()
            decoded = SimpleTokenizer().decode(caption)
            print(decoded)

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[i])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


def create_input_files(karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder, data_name,
                       max_len=256, seed_val=8008, max_vocab_size=11000):
    """
    Creates input files for training, validation, and test data.
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as file:
        data = json.load(file)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    tokenizer = SimpleTokenizer()

    for img in track(data, total=len(data), description="Processing tokens..."):

        captions = []
        for c in img['captions']:
            # Update word frequency
            captions.append(c)

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['file_path'])

        if "train" in path:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif "val" in path:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif "test" in path:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(seed_val)
    iterable = [(train_image_paths, train_image_captions, 'TRAIN'),
                (val_image_paths, val_image_captions, 'VAL'),
                (test_image_paths, test_image_captions, 'TEST')]

    for impaths, imcaps, split in iterable:

        h5_path = os.path.join(output_folder, split + '_IMAGES_' + data_name + '.hdf5')

        # remove h5 if already present
        if os.path.isfile(h5_path):
            os.remove(h5_path)

        with h5py.File(h5_path, 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            enc_captions = []
            caplens = []

            for i, path in track(enumerate(impaths), description=f"Creating {split} h5...", total=len(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)

                img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                img = img.transpose(2, 0, 1)

                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                #  encode every caption
                captions = ['<|startoftext|> ' + cap + ' <|endoftext|>' for cap in captions]
                caplens.append([len(cap) for cap in captions])
                enc_captions.append(tokenizer.tokenize(captions, context_length=max_len))

            # Sanity check
            print(f"{images.shape[0]}  == {len(enc_captions)} == {len(caplens)}")
            assert images.shape[0] == len(enc_captions) == len(caplens)

            enc_path = os.path.join(output_folder, split + '_CAPTIONS_' + data_name + '.pkl')
            caplens_path = os.path.join(output_folder, split + '_CAPLENS_' + data_name + '.json')

            # Save encoded captions and their lengths to JSON files
            with open(enc_path, 'wb') as file:
                pickle.dump(enc_captions, file)

            with open(caplens_path, 'w') as file:
                json.dump(caplens, file)


def preprocess_coco_ann(train_caption_ann, val_caption_ann, output_file):
    """
    Preprocess coco annotations for the create_input_file function
    :param train_caption_ann:
    :param val_caption_ann:
    :param output_file:
    :return:
    """
    val = json.load(open(val_caption_ann, 'r'))
    train = json.load(open(train_caption_ann, 'r'))

    # combine all images and annotations together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    # create the json blob
    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']

        # coco specific here, they store train/val images separately
        loc = 'train2017' if 'train' in img['coco_url'] else 'val2017'

        jimg = {}
        jimg['file_path'] = os.path.join(loc, img['file_name'])
        jimg['id'] = imgid

        sents = []
        annotsi = itoa[imgid]
        for a in annotsi:
            sents.append(a['caption'])
        jimg['captions'] = sents
        out.append(jimg)

    json.dump(out, open(output_file, 'w'))


if __name__ == '__main__':

    # change base_path depending where you have coco
    base_path = "/home/dizzi/Desktop/coco/"

    params = DataParams()

    # dependent paths
    ann_path = os.path.join(base_path, "annotations")
    train_caption_ann = os.path.join(ann_path, "captions_train2017.json")
    val_caption_ann = os.path.join(ann_path, "captions_val2017.json")
    karpathy_json_path = os.path.join(ann_path, "coco_raw.json")
    output_dir = os.path.join(base_path, "preprocessed")

    data_name = f"{params.captions_per_image}_cap_per_img_{params.min_word_freq}_min_word_freq"

    if not os.path.isfile(karpathy_json_path):
        preprocess_coco_ann(train_caption_ann, val_caption_ann, karpathy_json_path)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if False:
        create_input_files(karpathy_json_path=karpathy_json_path,
                           image_folder=base_path,
                           captions_per_image=params.captions_per_image,
                           min_word_freq=params.min_word_freq,
                           output_folder=output_dir,
                           max_len=params.TEXT_SEQ_LEN,
                           data_name=data_name,
                           max_vocab_size=params.vocab_size)

    train_data = CaptionDataset(output_dir, "TRAIN")
    val_data = CaptionDataset(output_dir, "VAL")

    t = train_data.__getitem__(3)
    v = val_data.__getitem__(1)
