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
from torch.utils.data import Dataset, DataLoader

from src.Parameters import DataParams, PathParams, DebugParams


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

        img_path = [elem for elem in files if "IMAGES" in elem]
        captions = [elem for elem in files if "CAPTIONS" in elem][0]
        caplens = [elem for elem in files if "CAPLENS" in elem][0]

        url_image = DataParams().generate_data_url

        if url_image:
            img_path = [elem for elem in img_path if "pkl" in elem][0]
            with open(os.path.join(data_folder, img_path), "rb") as file:
                self.imgs = pickle.load(file)
            self.cpi = self.imgs.pop(0)
        else:
            img_path = [elem for elem in img_path if "hdf5" in elem][0]
            h = h5py.File(os.path.join(data_folder, img_path), 'r')
            self.imgs = h['images']

            # Captions per image
            self.cpi = h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, captions), 'rb') as file:
            self.captions = pickle.load(file)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, caplens), 'rb') as file:
            self.caplens = pickle.load(file)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        random.seed(seed)

    def __getitem__(self, i):

        img = self.imgs[i]
        if isinstance(img, str):
            img = imread(img)
            img=img_processing(img)

        img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        cap_index = random.randint(0, len(self.captions[i]) - 1)
        caption = torch.LongTensor(self.captions[i][cap_index])
        caplen = torch.LongTensor([self.caplens[i][cap_index]])

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


def get_dataloaders(transform=None):
    pt_params = PathParams()
    db_params = DebugParams()

    # get dataloader
    train_data = CaptionDataset(pt_params.preprocessed_dir, "TRAIN", transform=transform)
    val_data = CaptionDataset(pt_params.preprocessed_dir, "VAL", transform=transform)

    train_dl = DataLoader(train_data, batch_size=db_params.batch_size, shuffle=True, drop_last=True,
                          num_workers=db_params.workers, pin_memory=db_params.pin_memory)
    val_dl = DataLoader(val_data, batch_size=db_params.batch_size, shuffle=True, drop_last=True,
                        num_workers=db_params.workers, pin_memory=db_params.pin_memory)

    return train_dl, val_dl


def img_processing(img):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = img.transpose(2, 0, 1)

    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255

    return img


def create_input_files(karpathy_json_path, image_folder, captions_per_image, output_folder, data_name,
                       max_len=256, seed_val=8008):
    """
    Creates input files for training, validation, and test data.
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length, also pad shorter sentences
    """

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as file:
        data = json.load(file)

    data_params = DataParams()

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

        if data_params.generate_data_url:
            path = img['url']
        else:
            path = os.path.join(image_folder, img['file_path'])

        if "train" in img['split']:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif "val" in img['split']:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif "test" in img['split']:
            test_image_paths.append(path)
            test_image_captions.append(captions)
        else:
            raise KeyError("Path not recognized")

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

        h5_path = split + '_IMAGES_' + data_name
        h5_path += '.pkl' if data_params.generate_data_url else ".hdf5"
        h5_path = os.path.join(output_folder, h5_path)

        # Create dataset inside HDF5 file to store images
        if data_params.generate_data_url:
            h = open(h5_path, "wb")
            pickle.dump(impaths, h)
        else:
            h = h5py.File(h5_path, 'w')
            dataset = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

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
            if not data_params.generate_data_url:
                img = imread(impaths[i])
                img=img_processing(img)

                # Save image to HDF5 file
                dataset[i] = img

            #  encode every caption
            captions = ['<|startoftext|> ' + cap + ' <|endoftext|>' for cap in captions]
            ec = tokenizer.tokenize(captions, context_length=max_len)
            enc_captions.append(ec)
            caplens.append([torch.count_nonzero(cap) for cap in ec])

        # Sanity check
        if data_params.generate_data_url:
            assert len(impaths) == len(enc_captions) == len(caplens)
        else:
            assert dataset.shape[0] == len(enc_captions) == len(caplens)

        h.close()

        enc_path = os.path.join(output_folder, split + '_CAPTIONS_' + data_name + '.pkl')
        caplens_path = os.path.join(output_folder, split + '_CAPLENS_' + data_name + '.pkl')

        # Save encoded captions and their lengths to JSON files
        with open(enc_path, 'wb') as file:
            pickle.dump(enc_captions, file)

        with open(caplens_path, 'wb') as file:
            pickle.dump(caplens, file)


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
        jimg['url'] = img['flickr_url']
        jimg['split'] = loc

        sents = []
        annotsi = itoa[imgid]
        for a in annotsi:
            sents.append(a['caption'])
        jimg['captions'] = sents
        out.append(jimg)

    json.dump(out, open(output_file, 'w'))


if __name__ == '__main__':

    # change base_path depending where you have coco

    data_params = DataParams()
    path_params = PathParams()

    # dependent paths
    ann_path = os.path.join(path_params.coco_path, "annotations")
    train_caption_ann = os.path.join(ann_path, "captions_train2017.json")
    val_caption_ann = os.path.join(ann_path, "captions_val2017.json")
    karpathy_json_path = os.path.join(ann_path, "coco_raw.json")

    data_name = f"{data_params.captions_per_image}_cap_per_img"

    if not os.path.isfile(karpathy_json_path):
        preprocess_coco_ann(train_caption_ann, val_caption_ann, karpathy_json_path)

    if not os.path.isdir(path_params.preprocessed_dir):
        os.mkdir(path_params.preprocessed_dir)

    if True:
        create_input_files(karpathy_json_path=karpathy_json_path,
                           image_folder=path_params.coco_path,
                           captions_per_image=data_params.captions_per_image,
                           output_folder=path_params.preprocessed_dir,
                           max_len=data_params.max_text_seq_len,
                           data_name=data_name,
                           )

    train_data = CaptionDataset(path_params.preprocessed_dir, "TRAIN")
    val_data = CaptionDataset(path_params.preprocessed_dir, "VAL")

    t = train_data.__getitem__(3)
    v = val_data.__getitem__(1)
