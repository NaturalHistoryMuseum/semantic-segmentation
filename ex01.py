#[1]
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

# rebuild train step by step
##import numpy as np
##import logging
##from torch import nn
##from torch import optim
##from torch.optim import lr_scheduler
##from segmentation.instances import SemanticLabels
##from matplotlib import gridspec
##
##def train_segmentation():
##  #pdb.set_trace()
##  cross_entropy = nn.CrossEntropyLoss(weight=train_loader.labelled.dataset.weights)
##  L2 = nn.MSELoss()
##  optimizer = optim.Adam(model.parameters(), lr=1e-3)
##  scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
##
##  losses = {'train': {'semantic': [], 'instance': [], 'total': []},
##                'test':  {'semantic': [], 'instance': [], 'total': []}}
##  accuracies = {'train': [], 'test': []}
##
##  for epoch in range(30):
##    scheduler.step()
##    if epoch % scheduler.step_size == 0:
##            logging.debug(f'Learning rate set to {scheduler.get_lr()[0]}')            
##    model.train()
##    for i in range(len(train_loader.labelled.dataset)):
##      training_data=train_loader.labelled.dataset[i]
##      labelled=isinstance(training_data, tuple)
##      if labelled:
##        print("epoch ", epoch, " sample " , i, " is a tuple")
##        image, labels, instances = training_data
##        #HACK: usue unsqueeze to correct expected stride...
##        #pdb.set_trace()
##        image, labels, instances = Variable(image.unsqueeze(0)), Variable(labels.unsqueeze(0)), Variable(instances.unsqueeze(0))
##      else:
##        print("epoch ", epoch, " sample " , i, " is not a tuple")
##      
##      optimizer.zero_grad()
##      z_hat1, x_hat, logits, instance_embeddings = model(image)
##      z1 = model.forward_clean(image)[0]
##      reconstruction_loss = L2(z_hat1, Variable(z1.data, requires_grad=False)) + L2(x_hat, image)
##      loss = 20 * reconstruction_loss
##      if labelled:
##        logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
##        semantic_loss = cross_entropy(logits_per_pixel.view(-1, 5), labels.view(-1))
##
##        instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
##                                        for embeddings, target_clusters
##                                        in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
##                                    for image_instance_embeddings, image_labels, image_instances
##                                    in torch_zip(instance_embeddings, labels, instances))
##
##        loss += semantic_loss * 10 + instance_loss
##
##        predicted_class = logits.data.max(1, keepdim=True)[1]
##        correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
##        accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)
##
##      loss.backward()
##      optimizer.step()
##
##      # losses['train']['semantic'].append(semantic_loss.item())
##      # losses['train']['instance'].append(instance_loss.item())
##      losses['train']['total'].append(loss.item())
##      # accuracies['train'].append(accuracy)
##      info = f'Epoch: {epoch + 1:{3}}, Batch: {i:{3}}, Loss: {loss.item()}'
##      if labelled:
##        info += f', Accuracy: {(accuracy * 100)}%'
##      logging.info(info)
##      
##      print(info)  
##    if (epoch + 1) % 1 == 0:
##      visualise_results(Path('models') / f'epoch_{epoch + 1}.png', image, x_hat, predicted_class,
##                            colours=train_loader.labelled.dataset.colours)
##      np.save('losses.npy', [{'train': losses['train'], 'test': losses['test']}])
##      np.save('accuracies.npy', [{'train': accuracies['train'], 'test': accuracies['test']}])
##
##    if (epoch + 1) % 2 == 0:
##      torch.save(model.state_dict(), Path('models') / f'epoch_{epoch + 1}')
##
##    
##    ##print(image)
##
##def torch_zip(*args):
##    for items in zip(*args):
##        yield tuple(item.unsqueeze(0) for item in items)
##
##def visualise_results(output, original_image, reconstructed_image, predicted_class, colours, dpi=500):
##    if not output.parent.exists():
##        output.parent.mkdir()
##
##    n = original_image.shape[0]
##    gs = gridspec.GridSpec(3, n, width_ratios=[2.42]*n, wspace=0.05, hspace=0)
##    plt.figure(figsize=(n * 2.42, 3))
##
##    for i in range(n):
##        plt.subplot(gs[0, i])
##        plt.imshow(original_image.data[i].cpu().numpy().transpose(1, 2, 0))
##        plt.axis('off')
##        plt.subplot(gs[1, i])
##        plt.imshow(reconstructed_image.data[i].cpu().numpy().transpose(1, 2, 0))
##        plt.axis('off')
##        plt.subplot(gs[2, i])
##        plt.imshow(visualise_segmentation(predicted_class[i], colours))
##        plt.axis('off')
##    plt.savefig(str(output), dpi=dpi, bbox_inches='tight')
##    plt.close('all')
##
##def visualise_segmentation(predicted_class, colours):
##    class_image = np.zeros((predicted_class.shape[1], predicted_class.shape[2], 3))
##    prediction = predicted_class[0].cpu().numpy()
##    for j in range(len(colours)):
##        class_image[prediction == j] = colours[j]
##    return class_image / 255

#[2]
model = SemanticInstanceSegmentation() #From network
instance_clustering = DiscriminativeLoss() #From instances
#[3]
transform = transforms.Compose([ #torchvision
    transforms.RandomRotation(5),
    transforms.RandomCrop((256, 768)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])

target_transform = transforms.Compose([transform, transforms.Lambda(lambda x: (x * 255).long())])

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

#import pdb; pdb.set_trace()
#train_segmentation()

#[4] Train
train(model, instance_clustering, train_loader, test_loader)

#[5] Evaluate
#model.load_state_dict(torch.load('models/epoch_20'))
