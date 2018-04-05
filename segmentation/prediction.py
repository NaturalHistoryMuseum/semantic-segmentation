import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from segmentation.datasets import ImageFolder
from segmentation.instances import mean_shift, visualise_instances
from segmentation.network import SemanticInstanceSegmentation

model = SemanticInstanceSegmentation().cuda()
model.load_state_dict(torch.load('models/epoch_30'))
model.eval()

data_unlabelled = ImageFolder(root='data/slides', transform=ToTensor())
loader = torch.utils.data.DataLoader(data_unlabelled, batch_size=1, shuffle=True)

image = F.pad(Variable(next(iter(loader))).cuda(), (0, 0, 3, 3))
_, logits, instance_embeddings = model.forward_clean(image)

current_logits = logits[0]

predicted_class = current_logits.data.max(0)[1]
predicted_instances = [None] * 5
for class_index in range(5):
    mask = predicted_class.view(-1) == class_index
    if mask.max() > 0:
        label_embedding = instance_embeddings[0].view(1, instance_embeddings.shape[1], -1)[..., mask]
        label_embedding = label_embedding.data.cpu().numpy()[0]

        predicted_instances[class_index] = mean_shift(label_embedding)

instance_image = visualise_instances(predicted_instances, predicted_class, num_classes=5)
num_instances = int(instance_image.max() + 1)

filtered_instance_image = np.zeros_like(instance_image)
for i in range(1, num_instances):
    instance = remove_small_objects(instance_image == i, min_size=128)
    if instance.sum() > 0:
        filtered_instance_image += i * instance

image = image[0].data.cpu().numpy().transpose(1, 2, 0)
regions = regionprops(filtered_instance_image.astype(np.int64))
plt.imsave('full.png', image)
for i, region in enumerate(regions):
    plt.imsave(f'{i}.png', image[region._slice])
