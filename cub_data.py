import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import datasets
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def get_CUB_raw():
    data_dir = Path("dataset")
    dataset = datasets.load_dataset("imagefolder", data_dir=data_dir / "CUB_200_2011" / "images")["train"]
    path2imageid = {}
    with open(data_dir / "CUB_200_2011" / "images.txt") as f:
        for line in f:
            id, path = line.strip().split()
            assert path not in path2imageid
            path2imageid[path] = id
    # train test split
    imageid2split = {}
    with open(data_dir / "CUB_200_2011" / "train_test_split.txt") as f:
        for line in f:
            # split: <is_training_image>
            image, split = line.strip().split()
            imageid2split[image] = ("train" if split == "1" else "test")

    # merge attributes
    attrid2label = {}
    with open(data_dir / "attributes.txt") as f:
        for line in f:
            id, label = line.strip().split()
            attrid2label[id] = label
    imageid2attr = defaultdict(list)
    with open(data_dir / "CUB_200_2011" / "attributes" / "image_attribute_labels.txt") as f:
        for line in f:
            id, attrid, is_present, *_ = line.strip().split()
            is_present = int(is_present)
            if is_present:
                imageid2attr[id].append(attrid2label[attrid])

    def add_metadata(example):
        # eg xxx/project/dataset/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg 
        # -> 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
        path = "/".join(example["image"].filename.split("/")[-2:])
        imageid = path2imageid[path]
        example["attributes"] = imageid2attr[imageid]
        example["split"] = imageid2split[imageid]
        return example

    dataset = dataset.map(add_metadata)

    train_ds = dataset.filter(lambda ex: ex["split"] == "train")
    test_ds = dataset.filter(lambda ex: ex["split"] == "test")

    dataset = datasets.DatasetDict({
        "train": train_ds,
        "test": test_ds
    })
    return dataset

def CUB_coarse():
    """
    Use coarse label (i.e. class name) such as "Black Footed Albatross"
    """
    def coarse_text(example):
        labels = example["label"]
        captions = []
        for label in labels:
            # e.g. 001.Black_footed_Albatross
            name = dataset["train"].features["label"].int2str(label)
            # convert to only Black Footed Albatross
            name = " ".join(name.split(".")[1].split("_"))
            captions.append(name)
        example["caption"] = captions
        return example

    dataset = get_CUB_raw()
    return dataset.map(coarse_text, batched=True)

def CUB_fine():
    """
    Use fine label (i.e. class name + attributes)
    In CUB dataset attributes are like 'has_back_color::grey' where has_back_color is base attributes and grey is value
    The fine caption will be <class name>, <attr1>, <attr2>, ...
    """
    dataset = get_CUB_raw()

    # 1. find most frequent 10 base attribute
    attr_counter = Counter()
    attr_base2value = defaultdict(Counter)
    for example in dataset["train"]:
        for attr in example["attributes"]:
            base_attr, value = attr.split("::")
            attr_counter[base_attr] += 1
            attr_base2value[base_attr][value] += 1
    include_base_attrs = set([attr for attr, _ in attr_counter.most_common(10)])

    # 2. for each of 10 base attribute, use most frequent 5 values
    include_attrs = {f"{base_attr}::{value}" 
                        for base_attr, value_counter in attr_base2value.items()
                        for value, _ in value_counter.most_common(5)
                        if base_attr in include_base_attrs}
    assert len(include_attrs) == 50

    # 3. convert to text, by simply concat
    def fine_text(example):
        # e.g. 001.Black_footed_Albatross
        name = dataset["train"].features["label"].int2str(example['label'])
        # convert to only Black Footed Albatross
        name = " ".join(name.split(".")[1].split("_"))
        # add attributes
        for attr in example["attributes"]:
            if attr not in include_attrs:
                continue
            # e.g. 'has_primary_color::brown'
            # convert to "brown primary color"
            assert attr.startswith("has_")
            attr_detail, attr_value = attr.split("::")
            attr = attr_value + " " + " ".join(attr_detail.split("_")[1:])
            name += ", " + attr
        example["caption"] = name
        return example
    return dataset.map(fine_text, batched=False)