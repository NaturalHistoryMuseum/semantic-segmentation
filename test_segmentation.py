#script that runs segmentation over a given set of unlabelled data
#[1] Libraries needed
from pathlib import Path
import random
import matplotlib.image as image_mgr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from segmentation.datasets import Slides, ImageFolder, SemiSupervisedDataLoader
from segmentation.instances import DiscriminativeLoss, mean_shift, visualise_embeddings, visualise_instances
from segmentation.network import SemanticInstanceSegmentation
from segmentation.training import train

# this produces the segmented images from unlabelled images as in jupyter example
def segment_this(model, filename):
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
        #[9] save result
   classes=Path(Path(filename).parent,str(Path(filename).stem+"_labels.png"))
   print(classes)
   image_mgr.imsave(classes, predicted_class.cpu().numpy())
   instances=Path(Path(filename).parent,str(Path(filename).stem+"_instances.png"))
   print(instances)
   image_mgr.imsave(instances, visualise_instances(predicted_instances, predicted_class, num_classes=5))
    

#[2] create model and instance cluster
model = SemanticInstanceSegmentation() #From network

#[3] Evaluate ** need to load model to evaluate
model.load_state_dict(torch.load('models/epoch_42'))
model.eval()

#[4]Evaluate on non-segmented images
# preprocessing: 
# - borders added
# - resized
# data/sheets/*.JPG

for filename in Path('data', 'sheets01').iterdir():
  if not "labels" in filename.name and not "instances" in filename.name and not "classes" in filename.name:
    segment_this(model, filename)
    #break
