#[1] Libraries needed
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

#[2] create model and instance cluster
model = SemanticInstanceSegmentation() #From network
instance_clustering = DiscriminativeLoss() #From instances

#[3] build transforms and initialise data sets 
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

#[4] Train ** No need to train now
#train(model, instance_clustering, train_loader, test_loader)

#[5] Evaluate ** need to evaluate to get model loaded
model.load_state_dict(torch.load('models/epoch_20'))
model.eval()

train_loader = torch.utils.data.DataLoader(train_data_labelled, batch_size=1, shuffle=True)

image, labels, instances = next(iter(train_loader))

image = Variable(image)
instances = Variable(instances + 1)
_, logits, instance_embeddings = model.forward_clean(image)

current_logits = logits[0]
current_labels = labels[0, 0]
current_instances = instances[0]

predicted_class = current_logits.data.max(0)[1]
predicted_instances = [None] * 5
for class_index in range(5):
    mask = predicted_class.view(-1) == class_index
    if mask.max() > 0:
        label_embedding = instance_embeddings[0].view(1, instance_embeddings.shape[1], -1)[..., mask]
        label_embedding = label_embedding.data.cpu().numpy()[0]

        predicted_instances[class_index] = mean_shift(label_embedding)

#[6] Visualise training ** need to visualise training results to get colors right

plt.rcParams['image.cmap'] = 'Paired'

fig, axes = plt.subplots(3, 2, figsize=(15, 10))
for ax in axes.flatten(): ax.axis('off')

axes[0, 0].set_title('Original image')
axes[0, 0].imshow(image[0].data.numpy().transpose(1, 2, 0))
axes[1, 0].set_title('Ground truth classes')
axes[1, 0].imshow(current_labels.cpu().numpy().squeeze())
axes[2, 0].set_title('Ground truth instances')
axes[2, 0].imshow(current_instances.cpu().numpy().squeeze())
axes[1, 1].set_title('Predicted classes')
axes[1, 1].imshow(predicted_class.cpu().numpy().squeeze())
instance_image = visualise_instances(predicted_instances, predicted_class, num_classes=5)
axes[2, 1].set_title('Predicted instances')
axes[2, 1].imshow(instance_image)

#[7] Explicitly clear GPU **

del (logits, instance_embeddings, instance_image, image, labels,
     instances, current_logits, current_labels, current_instances,
     mask, label_embedding, predicted_class, predicted_instances)

#[8]Evaluate on full image
#1 data/slides_subset/010646725_816445_1431072.JPG
#2 data/slides_subset/010646726_816445_1431072.JPG
#3 data/slides_subset/010646727_816445_1431072.JPG
for filename in Path('data', 'slides_subset').iterdir():
    response = input(str('try:'+ filename.name + '?'))
    if response == 'y':
        print("processing", filename.name)
        image = torch.Tensor((plt.imread(filename) / 255).transpose(2, 0, 1)).unsqueeze(0)
        _, logits, instance_embeddings = model.forward_clean(image)
        predicted_class = logits[0].data.max(0)[1]
        instance_embeddings = instance_embeddings[0]
    
        predicted_instances = [None] * 5
        for class_index in range(5):
            mask = predicted_class.view(-1) == class_index
            if mask.max() > 0:
               label_embedding = instance_embeddings.view(1, instance_embeddings.shape[0], -1)[..., mask]
               label_embedding = label_embedding.data.cpu().numpy()[0]

               predicted_instances[class_index] = mean_shift(label_embedding)

        #[9] Visualise result
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        for ax in axes: ax.axis('off')
    
        axes[0].set_title('Original image')
        axes[0].imshow(image[0].numpy().transpose(1, 2, 0))
    
        axes[1].set_title('Predicted classes')
        axes[1].imshow(predicted_class.cpu().numpy())
    
        axes[2].set_title('Predicted instances')
        axes[2].imshow(visualise_instances(predicted_instances, predicted_class, num_classes=5))
    
        plt.show()
    else:
        print("skip ", filename.name )
