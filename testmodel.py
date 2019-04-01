# code imported from jupiter notebook
#[1] Required libraries
from itertools import islice
import logging

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from segmentation.datasets import Slides, ImageFolder, SemiSupervisedDataLoader
from segmentation.instances import DiscriminativeLoss, mean_shift, visualise_embeddings, visualise_instances
from segmentation.network import SemanticInstanceSegmentation
from segmentation.training import train

#[2] create model and clustening function
model = SemanticInstanceSegmentation() #From network
instance_clustering = DiscriminativeLoss() #From instances

#[3] random transforms for pictures
#**************************************************
#convert to parameters random crop heigth and width
#**************************************************
transform = transforms.Compose([ #torchvision
    transforms.RandomRotation(5),
    transforms.RandomCrop((256, 768)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])

target_transform = transforms.Compose([transform, transforms.Lambda(lambda x: (x * 255).long())])

#**************************************************
#convert to parameter batch_size
#**************************************************
batch_size = 3

# WARNING: Don't use multiple workers for loading! Doesn't work with setting random seed
# Slides: copies the data if required into the data/raw/[images,
# instances, labels] directories and returns
# import pdb; pdb.set_trace()
train_data_labelled = Slides(download=True, train=True, root='data', transform=transform, target_transform=target_transform)
train_loader_labelled = torch.utils.data.DataLoader(train_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_data_unlabelled = ImageFolder(root='data/slides', transform=transform)
train_loader_unlabelled = torch.utils.data.DataLoader(train_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
train_loader = SemiSupervisedDataLoader(train_loader_labelled, train_loader_unlabelled)

test_data_labelled = Slides(download=True, train=False, root='data', transform=transform, target_transform=target_transform)
test_loader_labelled = torch.utils.data.DataLoader(test_data_labelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_data_unlabelled = ImageFolder(root='data/slides', transform=transform)
test_loader_unlabelled = torch.utils.data.DataLoader(test_data_unlabelled, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = SemiSupervisedDataLoader(test_loader_labelled, test_loader_unlabelled)


#[4] test model


losses = {'train': {'semantic': [], 'instance': [], 'total': []},
              'test':  {'semantic': [], 'instance': [], 'total': []}}

accuracies = {'train': [], 'test': []}

#**************************************************
#convert to parameter epoch to test
#**************************************************
epoch = 6

model.load_state_dict(torch.load('models/epoch_6'))
model.eval()
total_loss = 0
total_accuracy = 0
        
num_test_batches = 1
        
with torch.no_grad():
    for image, labels, instances in islice(test_loader, num_test_batches):
        image, labels, instances = (Variable(tensor) for tensor in (image, labels, instances))
        
        z_hat1, x_hat, logits, instance_embeddings = model(image)
        z1 = model.forward_clean(image)[0]
        # logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
        # semantic_loss = cross_entropy(logits_per_pixel.view(-1, 5), labels.view(-1))
        #
        # instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
        #                         for embeddings, target_clusters
        #                         in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
        #                     for image_instance_embeddings, image_labels, image_instances
        #                     in torch_zip(instance_embeddings, labels, instances))
        #
        # loss = semantic_loss * 10 + instance_loss
        loss = L2(z_hat1, z1) + L2(x_hat, image)
        
        total_loss += loss.item()
        
        predicted_class = logits.data.max(1, keepdim=True)[1]
        correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
        accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)
        total_accuracy += accuracy
        
    average_loss = total_loss / num_test_batches
    average_accuracy = total_accuracy / num_test_batches
    losses['test']['total'].append(average_loss)
    accuracies['test'].append(average_accuracy)
    logging.info(f'Epoch: {epoch + 1:{3}}, Test Set, Cross-entropy loss: {average_loss}, Accuracy: {(average_accuracy * 100)}%')
