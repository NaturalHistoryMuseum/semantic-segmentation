import numpy as np
from PIL import Image

colours = [np.array([0, 0, 0]), np.array([255, 255, 255]), np.array([255,   0,   0]), np.array([255, 255,   0])]

values = ((np.array(colours) // 255) * np.array([1, 2, 4]).reshape(1, 3)).sum(axis=1)

image =Image.open('example_classes.png')
image1 =Image.open('herb_ex_labels.png')
image2 =Image.open('BM000075813_labels.png')

img = image1
img = 1 * (np.asarray(img) > 128)

image_colours = (img * np.array([1, 2, 4]).reshape(1, 1, 3)).sum(axis=2)
