# EmergentImagination

To install use the setup file as follows:
```
python setup.py install
```

If you wish to play around with the repo and change some code use the `-e` option as follows:

```
python setup.py develop

```


## Dataset 
This project uses [COCO2017](https://cocodataset.org/#download). 
There are two ways to train the model, for both you need to download the [annotation file](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) from the COCO website.


#### 1.Local Train
This kind of train requires you to have Download the [train](http://images.cocodataset.org/zips/train2017.zip) (18GB)
and [val](http://images.cocodataset.org/zips/val2017.zip) (1GB) split on your machine.

Once downloaded be sure to mimic the following directory structure
```
coco
    ├── annotations
    │   └── deprecated-challenge2017
    ├── train2017
    └── val2017
```



#### 2.Remote train
This training instance does not require the dataset to be accessible on your local machine. The images are extracted directly from the coco website, so you need a stable connection.

### Dataset generation
Once you chose either one of training instances you need to create the appropriate files.
We use the [sgrvinod tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) for the `.hdf5` generation.

Simply process the dataset with:
```
python dataset.py --coco_path /path/to/coco 
```
For the first instance and 

```
python dataset.py --coco_path /path/to/coco --generate_data_url True
```
For the second one.

You can find the above-mentioned parameters in the [Parameters](./src/Parameters.py) class.

Once done you can find a new directory called `Preprocessed` containing the required files for training. 

## Receiver Model
Receiver model taken from [here](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py)
Wordmap and model checkpoint can be found [here](https://drive.google.com/drive/folders/189VY65I_n4RTpQnmLGj7IzVnOF6dmePC)


## Sender model 
Taken from [here](https://github.com/lucidrains/DALLE-pytorch)
