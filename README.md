# Semi-supervised semantic and instance segmentation

[Image segmentation](https://en.wikipedia.org/wiki/Image_segmentation) is a process that breaks down an image into smaller segments according to some criteria. For _semantic_ segmentation this means that all pixels of one segment represent an object (or objects) of a single _class_; in this case a class has been predefined to represent a type of object of interest. For example in the slide image shown below, the objects can be classified as being a specimen (in the centre), regular labels (either side of the specimen), type labels (red circle), barcode labels, or otherwise as being part of the 'background'.

![slide image](https://github.com/NaturalHistoryMuseum/semantic-segmentation/blob/master/example_image.JPG)

Segments can be broken down further into _instances_, which represent distinct objects separately - even if they have the same class. For example, the labels on either side of the slide image would be part of the same 'segment' (even though it is not contiguous), but are treated as separate instances. A corresponding representation of instances for the slide image can be seen below, where each unique colour maps pixels to a specific instance.

![slide image](https://github.com/NaturalHistoryMuseum/semantic-segmentation/blob/master/example_instances.png)

In situations where labelled example data is hard to come by, as is the case for manually generated example segmentations for slides, it is desirable to use methods that can perform well with small datasets. [Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning) covers a number of techniques to leverage large datasets of _unlabelled_ data to enhance the capability of models otherwise learned on small amounts of data.

This project combines approaches from each of these problems, primarily using the following prior works as reference:
* [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
* [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551)
* [Semi-Supervised Learning with Ladder Networks](https://arxiv.org/abs/1507.02672)

## Installation
The implementation is entirely in Python and all dependencies can be installed most straightforwardly by using the [Anaconda](https://anaconda.org/) package manager. Creating a new conda environment is done using the command:
```
conda env create -f environment.yml
```
which will create a new environment with the name `segmentation`. It is assumed that a PyTorch-supported GPU is available; notices of dropping support for GPU models can be found in the PyTorch [release notes](https://github.com/pytorch/pytorch/releases). For older GPUs it may still be possible to install PyTorch from source although behaviour may not be guaranteed. Adapting the code to work without a GPU should be straightforward but could be prohibitively slow - especially for training.

## Usage
Notes on usage can be found in the [wiki](https://github.com/NaturalHistoryMuseum/semantic-segmentation/wiki/Training-and-predicting-on-new-data)
