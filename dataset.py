"""The Dataset format of CUB"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import skimage


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths]) ## is train in the pkl_file_paths (['train.pkl', 'test.pkl'])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths]) ## if false, is test or val in pkl_file_paths
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))    ## open train.pkl, test.pkl and val.pkl and load the data to self.data
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img    ## wheather to load images. In some cases we don't need to load images since only the classification of the concepts need to be execute.
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path'].replace("\\", '/')
        try:
            img_path = self.image_dir + '/' + img_path

            img = skimage.io.imread(img_path)
            img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range

            if img.ndim == 3:                 # color image HWC
                img = img.mean(axis=-1)       # or use luminance: img = img[..., :3] @ [0.299,0.587,0.114]
            elif img.ndim == 2:               # already grayscale HW
                pass
            else:
                raise ValueError(f"Unexpected shape {img.shape}")
            img = img[None, ...] 
            
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:

            attr_label = img_data['attribute_label']
            
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((6, self.n_class_attr))
                    one_hot_attr_label[np.arange(6), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label



def load_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, n_class_attr=2, image_dir='images'):

    is_training = any(['train.pkl' in f for f in pkl_paths])

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform)
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)