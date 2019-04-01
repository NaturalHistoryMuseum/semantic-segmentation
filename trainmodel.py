# code imported from jupiter notebook
#[1] Required libraries
from pathlib import Path
import random
import configparser

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from segmentation.datasets import HerbariumSheets, ImageFolder, SemiSupervisedDataLoader
from segmentation.instances import DiscriminativeLoss, mean_shift, visualise_embeddings, visualise_instances
from segmentation.network import SemanticInstanceSegmentation
from segmentation.training import train

#[2] create model and clustening function
#**************************************************
# extracted label classes as parameters
#**************************************************
#read initial values from segmentation.ini
source_dir = 'nmwherbarium'
ini_file = Path().absolute().parent / source_dir / "segmentation.ini"
if ini_file.exists():
    seg_config = configparser.ConfigParser()
    seg_config.read(ini_file)
    # read rotation value
    random_rotation = int(seg_config['DEFAULT']["randomrotation"])
    # read number of labelling classes
    label_classes = int(seg_config['DEFAULT']["labelclasses"])
    # read cropping values
    crop_height  = int(seg_config['DEFAULT']["cropheight"])
    crop_width = int(seg_config['DEFAULT']["cropwidth"])
    # default batch size
    batch_size = int(seg_config['DEFAULT']["batchsize"])
    # read number of labelling classes
    label_classes = int(seg_config['DEFAULT']["labelclasses"])
    # read number of epochs to train for
    epochs = int(seg_config['DEFAULT']["trainepochs"])
else:
    # default rotation value
    random_rotation = 5
    # default values for slides
    label_classes = 5 
    crop_height  = 256
    crop_width = 768
    # default batch size
    batch_size = 3
    # default number labelling of classes
    label_classes = 5
    # default number of epochs to train for
    epochs = 40
    # default batch size
    batch_size = 3

# set the number of label classes
model = SemanticInstanceSegmentation(label_classes) #From network
instance_clustering = DiscriminativeLoss() #From instances

#[3] random transforms for pictures
transform = transforms.Compose([ #torchvision
    transforms.RandomRotation(random_rotation),
    transforms.RandomCrop((crop_height, crop_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])

target_transform = transforms.Compose([transform, transforms.Lambda(lambda x: (x * 255).long())])


# WARNING: Don't use multiple workers for loading! Doesn't work with setting random seed
# Slides: copies the data if required into the data/raw/images,
# Slides, labels] directories and returns
# import pdb; pdb.set_trace()
train_data_labelled = HerbariumSheets(download=True, train=True, root='data', transform=transform, target_transform=target_transform,images_dir = source_dir)
train_loader_labelled = torch.utils.data.DataLoader(train_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_data_unlabelled = ImageFolder(root='data/sheets', transform=transform)
train_loader_unlabelled = torch.utils.data.DataLoader(train_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_loader = SemiSupervisedDataLoader(train_loader_labelled, train_loader_unlabelled)

test_data_labelled = HerbariumSheets(download=True, train=False, root='data', transform=transform, target_transform=target_transform,images_dir = source_dir)
test_loader_labelled = torch.utils.data.DataLoader(test_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_data_unlabelled = ImageFolder(root='data/sheets', transform=transform)
test_loader_unlabelled = torch.utils.data.DataLoader(test_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = SemiSupervisedDataLoader(test_loader_labelled, test_loader_unlabelled)


#[4] Train model
train(model, instance_clustering, train_loader, test_loader, epochs, label_classes)


