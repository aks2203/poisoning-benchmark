"""
    The following class is heavily based on code by Meng Lee, mnicnc404. Date: 2018/06/04
    via
    https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
"""
import glob
import os

import torch
import numpy as np
from PIL import Image


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Author: Meng Lee, mnicnc404
    Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    """

    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = "val_annotations.txt"
    CLASSES = "words.txt"

    def __init__(self, root, split="train", transform=None, classes="all"):
        """Init with split, transform, target_transform. use --cached_dataset data is to be kept in memory."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = None
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self.split_dir, "**", "*.%s" % self.EXTENSION),
                recursive=True,
            )
        )
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        # build class label - number mapping
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}
        if self.split == "train":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == "val":
            with open(
                os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), "r"
            ) as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]
        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(root, self.CLASSES), "r") as file:
            for line in file:
                label_text, word = line.split("\t")
                label_text_to_word[label_text] = word.split(",")[0].rstrip("\n")
        self.classes = [label_text_to_word[label] for label in self.label_texts]
        self.targets = [
            self.labels[os.path.basename(file_path)] for file_path in self.image_paths
        ]
        if classes == "firsthalf":
            idx = np.where(np.array(self.targets) < 100)[0]
        elif classes == "lasthalf":
            idx = np.where(np.array(self.targets) >= 100)[0]
        else:
            idx = np.arange(len(self.targets))
        self.image_paths = [self.image_paths[i] for i in idx]
        self.targets = [self.targets[i] for i in idx]
        self.targets = [t - min(self.targets) for t in self.targets]

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return  image and label"""
        file_path = self.image_paths[index]
        img = Image.open(file_path)
        img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        if self.split == "test":
            return img
        else:
            return img, self.targets[index]
