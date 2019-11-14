# code imported from jupiter notebook
#[1] Required libraries
from pathlib import Path
import random
import configparser
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from segmentation.datasets import SpecimenImages, ImageFolder, SemiSupervisedDataLoader
from segmentation.instances import DiscriminativeLoss, mean_shift, visualise_embeddings, visualise_instances
from segmentation.network import SemanticInstanceSegmentation
from segmentation.training import train

#[2] create model and clustening function
#**************************************************
# extracted label classes as parameters
#**************************************************
#read initial values from segmentation.ini
source_dir = 'slides/rbgkslides'
ini_file = Path().absolute().parent / source_dir / "segmentation.ini"
unlabelled_dir = Path().absolute().parent / source_dir / "unlabelled"
if ini_file.exists():
    seg_config = configparser.ConfigParser()
    seg_config.read(ini_file)
    # read values from ini file
    # number of labelling classes
    label_classes = int(seg_config['DEFAULT']["labelclasses"])
    # rotation (for random rotation)
    random_rotation = int(seg_config['DEFAULT']["randomrotation"])
    # height and width (for random cropping)
    crop_height  = int(seg_config['DEFAULT']["cropheight"])
    crop_width = int(seg_config['DEFAULT']["cropwidth"])
    # batch size
    batch_size = int(seg_config['DEFAULT']["batchsize"])
    # number of epochs to train for
    epochs = int(seg_config['DEFAULT']["trainepochs"])
else:
    # use default values for slides
    # default values for slides
    label_classes = 5 
    random_rotation = 5  # rotation (for random rotation)
    crop_height  = 256   # height and width (for random cropping)
    crop_width = 768
    batch_size = 3       # batch size
    epochs = 40 # number of epochs to train for

if unlabelled_dir.exists():
    #copy the unlabelled images to data/unlabelled
    dest_dir = Path("data").expanduser()/'unlabelled'
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filepath in sorted(unlabelled_dir.glob('*')):
         shutil.copy(str(filepath), Path(dest_dir, filepath.name)) 
    
# set the number of label classes
model = SemanticInstanceSegmentation(label_classes).cuda() 
instance_clustering = DiscriminativeLoss().cuda()

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
train_data_labelled = SpecimenImages(download=True, train=True, root='data', transform=transform, target_transform=target_transform,images_dir = source_dir)
train_loader_labelled = torch.utils.data.DataLoader(train_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_data_unlabelled = ImageFolder(root='data/unlabelled', transform=transform)
train_loader_unlabelled = torch.utils.data.DataLoader(train_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_loader = SemiSupervisedDataLoader(train_loader_labelled, train_loader_unlabelled)

test_data_labelled = SpecimenImages(download=True, train=False, root='data', transform=transform, target_transform=target_transform,images_dir = source_dir)
test_loader_labelled = torch.utils.data.DataLoader(test_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_data_unlabelled = ImageFolder(root='data/unlabelled', transform=transform)
test_loader_unlabelled = torch.utils.data.DataLoader(test_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = SemiSupervisedDataLoader(test_loader_labelled, test_loader_unlabelled)


#[4] Train model
train(model, instance_clustering, train_loader, test_loader, epochs, label_classes)
