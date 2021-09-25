import json
import os
from collections import Counter
from random import seed, choice, sample

import cv2
import h5py
import nltk
import numpy as np
import torch
from imageio import imread
from rich.progress import track
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


def create_input_files( karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100, seed_val=8008):
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
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()


    for img in track(data, total=len(data), description="Processing tokens..."):
        captions = []
        for c in img['captions']:
            # Update word frequency
            tokens = nltk.word_tokenize(c)
            word_freq.update(tokens)
            if len(tokens) <= max_len:
                captions.append(tokens)

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
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(seed_val)
    iterable= [(train_image_paths, train_image_captions, 'TRAIN'),
               (val_image_paths, val_image_captions, 'VAL'),
               (test_image_paths, test_image_captions, 'TEST')]

    for impaths, imcaps, split in iterable:

        h5_path = os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5')

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

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


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

    # dependent paths
    ann_path = os.path.join(base_path, "annotations")
    train_caption_ann = os.path.join(ann_path, "captions_train2017.json")
    val_caption_ann = os.path.join(ann_path, "captions_val2017.json")
    karpathy_json_path = os.path.join(ann_path, "coco_raw.json")
    output_dir = os.path.join(base_path, "preprocessed")

    if not os.path.isfile(karpathy_json_path):
        preprocess_coco_ann(train_caption_ann, val_caption_ann, karpathy_json_path)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    create_input_files(karpathy_json_path=karpathy_json_path,
                       image_folder=base_path,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=output_dir,
                       max_len=50)
