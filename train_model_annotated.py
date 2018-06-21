# [1] Semantic and Instance segementation
# Combined approach to doing semantic and instance segmentation of images.
# This means segmenting the image such that each segment represents a unique    
# object (an instance) and that object is mapped to a (semantic) class.
# In the slides example, the classes are: background, specimens, labels,
# barcodes, and type labels
##
## Required libraries
##
from pathlib import Path # access the file system
import random            # provide a random generator

import matplotlib.pyplot as plt  # for displaying results
import torch                     # machine learning library
import torch.nn.functional as F  # set of convolution functions on neural networks
from torch.autograd import Variable # A tensor wrapper that records operations applied to it.
from torchvision import transforms #Common image transforms. They can be chained together using "Compose"

# Semantic Segmentation libraries, form James Durrant
from segmentation.datasets import Slides, ImageFolder, SemiSupervisedDataLoader
from segmentation.instances import DiscriminativeLoss, mean_shift, visualise_embeddings, visualise_instances
from segmentation.network import SemanticInstanceSegmentation
from segmentation.training import train

#uncomment if debugging
#import pdb; pdb.set_trace() 

#[2] Define model
# The model is a neural network with two heads: one for the semantic class
# embeddings, and one for the instance embedding.
# A discriminative loss function is used that encourages embeddings from the
# same instance to be closer to each other than to embedd to other instances

##
##Initialise model and clustering function
##
model = SemanticInstanceSegmentation() #From network
instance_clustering = DiscriminativeLoss() #From instances

#[3] Load data
transform = transforms.Compose([ #torchvision
    transforms.RandomRotation(5),
    transforms.RandomCrop((256, 768)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])

target_transform = transforms.Compose([transform, transforms.Lambda(lambda x: (x * 255).long())])

batch_size = 3

# WARNING: Don't use multiple workers for loading! Doesn't work with setting random seed

# Slides: copies the data if required into the data/raw/[images, instances,
# labels] directories and returns
# import pdb; pdb.set_trace() #uncomment if debugging
train_data_labelled = Slides(download=True, train=True, root='data', transform=transform, target_transform=target_transform)
train_loader_labelled = torch.utils.data.DataLoader(train_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
# how many slides to  use?
train_data_unlabelled = ImageFolder(root='data/slides', transform=transform)
train_loader_unlabelled = torch.utils.data.DataLoader(train_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_loader = SemiSupervisedDataLoader(train_loader_labelled, train_loader_unlabelled)

# is it ok for test dataset to be the same as the training dataset?
test_data_labelled = Slides(download=True, train=False, root='data', transform=transform, target_transform=target_transform)
test_loader_labelled = torch.utils.data.DataLoader(test_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_data_unlabelled = ImageFolder(root='data/slides', transform=transform)
test_loader_unlabelled = torch.utils.data.DataLoader(test_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = SemiSupervisedDataLoader(test_loader_labelled, test_loader_unlabelled)

#[4] Train
train(model, instance_clustering, train_loader, test_loader)

#[5] Evaluate
#model.load_state_dict(torch.load('models/epoch_20'))
