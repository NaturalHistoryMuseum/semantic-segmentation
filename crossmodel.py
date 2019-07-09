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

from segmentation.datasets import SpecimenImages, ImageFolder, SemiSupervisedDataLoader
from segmentation.instances import DiscriminativeLoss, mean_shift, visualise_embeddings, visualise_instances
from segmentation.network import SemanticInstanceSegmentation
from segmentation.training import train, evaluateepochs


#[2] read initial values from segmentation.ini
source_dir = 'herbarium/nmwhs_01'
ini_file = Path().absolute().parent / source_dir / "segmentation.ini"
unlabelled_dir = Path().absolute().parent / source_dir / "unlabelled"

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

#[3] create model and clustening function
model = SemanticInstanceSegmentation(label_classes).cuda()
instance_clustering = DiscriminativeLoss().cuda()

#[4] set random transforms for pictures
transform = transforms.Compose([ #torchvision
    transforms.RandomRotation(5),
    transforms.RandomCrop((1728, 1152)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])

target_transform = transforms.Compose([transform, transforms.Lambda(lambda x: (x * 255).long())])

# WARNING: Don't use multiple workers for loading! Doesn't work with setting random seed

test_data_labelled = SpecimenImages(download=False, train=False, root='data', transform=transform, target_transform=target_transform,images_dir = source_dir)
test_loader_labelled = torch.utils.data.DataLoader(test_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_data_unlabelled = ImageFolder(root='data/unlabelled', transform=transform)
test_loader_unlabelled = torch.utils.data.DataLoader(test_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = SemiSupervisedDataLoader(test_loader_labelled, test_loader_unlabelled)

#[5] Train model
epochs_dir = '../model01/models/'
evaluateepochs(model, instance_clustering, test_loader, epochs, epochs_dir)
