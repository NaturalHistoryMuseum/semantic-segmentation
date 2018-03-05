from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from skimage.measure import regionprops
import skimage
import skimage.io
import skimage.morphology
import skimage.transform
import torch
import torch.nn.functional as F
from torchvision import transforms

from datasets import Slides
from network import SemanticSegmentation


def downsample_pad(image):
    width, height = image.size
    if height > width:
        return downsample_pad(image.rotate(90))

    new_height = (height * 1024) // width
    image = image.resize((1024, new_height))

    padding_top = (512 - new_height) // 2
    padding_bottom = 512 - new_height - padding_top

    tensor = transforms.functional.to_tensor(image).unsqueeze(0)

    return F.pad(tensor, (0, 0, padding_top, padding_bottom)), (width, height), (padding_top, padding_bottom)


def color_region(region, color_image):
    return color_image[region._slice] * region.image[..., np.newaxis]


data = Slides(download=True, train=True, root='data')

model = SemanticSegmentation().cuda()
model.load_state_dict(torch.load(Path('models') / 'epoch_3200'))
model.eval()

input_folder = Path().absolute().parent / 'LouseDigitisation_LowResBackup'

for filename in input_folder.glob('**/*.JPG'):
    original_image = Image.open(filename)
    image, original_size, padding = downsample_pad(original_image)

    with torch.no_grad():
        logits = model(image.cuda())
        predicted_class = F.pad(logits.data.max(1, keepdim=True)[1], (0, 0, -padding[0], -padding[1]))

    upsampled_classes = skimage.transform.resize(predicted_class.data.cpu().numpy().astype(np.uint8)[0, 0], original_size[::-1], order=0)
    labels = skimage.img_as_ubyte(upsampled_classes)
    connected_labels = skimage.morphology.label(labels)
    regions = regionprops(connected_labels, intensity_image=labels)
    segments = ((color_region(region, np.asarray(original_image)), data.idx_to_class[region.max_intensity])
                for region in regions if region.area / labels.size > 0.005)

    output_folder = Path('segments') / filename.stem
    output_folder.mkdir(exist_ok=True, parents=True)

    shutil.copy(filename, output_folder / filename.parts[-1])
    for i, (segment, class_name) in enumerate(segments):
        segment = Image.fromarray(segment)
        segment.save(output_folder / f'{i}_{class_name}.jpg')
