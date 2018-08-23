import numpy as np
from PIL import Image

colours = [np.array([0, 0, 0]), np.array([255, 255, 255]), np.array([255,   0,   0]), np.array([255, 255,   0])]

values = ((np.array(colours) // 255) * np.array([1, 2, 4]).reshape(1, 3)).sum(axis=1)
##########################################
## image_colours = (img * np.array([1, 2, 4]).reshape(1, 1, 3)).sum(axis=2)
## ValueError: operands could not be broadcast together with shapes (1764,1169,4) (1,1,3) 
##########################################
# using image1 repliates the error

image =Image.open('example_classes.png')
image1 =Image.open('herb_ex_labels.png')
image2 =Image.open('BM000075813_labels.png')

img = image1
img = 1 * (np.asarray(img) > 128)

# the following line is the one that causes the error
# this is apparently due to the color mode in the PNG files (saved as RGBA),
# to correct reopened and saved all using PNG compression 8 (removes alpha channel).
image_colours = (img * np.array([1, 2, 4]).reshape(1, 1, 3)).sum(axis=2)
