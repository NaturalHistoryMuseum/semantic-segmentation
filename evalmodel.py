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
source_dir = 'slides/rgbkslides'
ini_file = Path().absolute().parent / source_dir / "segmentation.ini"
unlabelled_dir = Path().absolute().parent / source_dir / "unlabelled"

if ini_file.exists():
    seg_config = configparser.ConfigParser()
    seg_config.read(ini_file)
    # read values from ini file
    # number of labelling classes
    label_classes = int(seg_config['DEFAULT']["labelclasses"])
    # rotation value (for random rotation)
    random_rotation = int(seg_config['DEFAULT']["randomrotation"])
    # cropping height and width  (for random crop)
    crop_height  = int(seg_config['DEFAULT']["cropheight"])
    crop_width = int(seg_config['DEFAULT']["cropwidth"])
    # batch size
    batch_size = int(seg_config['DEFAULT']["batchsize"])
    # number of epochs to train for
    epochs = int(seg_config['DEFAULT']["trainepochs"])
else:
    # default values for slides
    label_classes = 5    # number of labelling of classes
    random_rotation = 5  # rotation value (for random rotation)
    crop_height  = 256   # crop height (for random crop) 
    crop_width = 768     # crop width (for random crop)
    batch_size = 3       # batch size
    epochs = 40          # number of epochs to train for

#[3] create model and clustening function
model = SemanticInstanceSegmentation(label_classes).cuda()
instance_clustering = DiscriminativeLoss().cuda()

#[4] set random transforms for pictures
transform = transforms.Compose([ #torchvision
    transforms.RandomRotation(random_rotation),
    transforms.RandomCrop((crop_height, crop_width)),
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
epochs_dir = 'models/'
evaluateepochs(model, instance_clustering, test_loader, epochs, epochs_dir)
