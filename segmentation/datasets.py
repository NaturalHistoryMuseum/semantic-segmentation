from itertools import chain, cycle
from pathlib import Path
import configparser

import random
import shutil
import tempfile
import urllib.request
import zipfile

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils import data

def identity(x):
    """To be used in place of explicitly checking whether there are transforms to be applied"""
    return x


class SemanticSegmentationDataset(data.Dataset):
    """
    raw_folder: the folder where the images are stored
    processed_folder: the folder where hdf5 files are stored
    training_file: the hdf5 file where training image data is stored
    test_file: the hdf5 file where test image data is stored
    """
    @property
    def raw_folder(self): return self.root / 'raw'

    @property
    def processed_folder(self): return self.root / 'processed'

    @property
    def training_file(self): return self.processed_folder / 'training.hdf5'

    @property
    def test_file(self): return self.processed_folder / 'test.hdf5'

    @property
    def idx_to_class(self): return sorted(self.class_to_idx, key=self.class_to_idx.get)

    def __init__(self, root, images_dir, train=True, transform=identity, target_transform=identity, download=False):
        
        self.root = Path(root).expanduser()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not (self.training_file.exists() and self.test_file.exists()):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.datafile = self.training_file if self.train else self.test_file
        self.class_to_idx, colours = self.read_label_file(self.processed_folder / 'label_colors.txt')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        with h5py.File(self.datafile, 'r') as f:
            img = f['images'][index]
            target = f['labels'][index]
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        target = Image.fromarray(target.astype(np.uint8), mode='L')

        seed = np.random.randint(2147483647)
        random.seed(seed)
        img = self.transform(img)
        random.seed(seed)
        target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.train_size if self.train else self.test_size

    def process_raw_image_files(self, folder_path, f_train, f_test, extension='*.JPG'):
        train_set = f_train.create_dataset('images', (self.train_size, self.height, self.width, 3), dtype=np.float32)
        test_set = f_test.create_dataset('images', (self.test_size, self.height, self.width, 3), dtype=np.float32)
        images = (Image.open(filename) for filename in sorted(folder_path.glob(extension)))
        j=k=l=0
        for i, image in enumerate(images):
            if j>self.train_size-1:
                #add it to the test set
                test_set[k] = np.asarray(image) / 255
                k+=1
            else:
                #add it to the train set
                train_set[l] = np.asarray(image) / 255
                l+=1
            j+=1
            if j>9:
                j=0

    def read_label_file(self, path):
        with open(path, 'r') as f:
            labels = f.read().strip().split('\n')
            colours = []
            class_to_idx = {}

            for i, label in enumerate(labels):
                *rgb, name = label.split()
                colours.append(np.array(list(map(int, rgb))))
                class_to_idx[name] = i

            return class_to_idx, colours

    def __repr__(self):
        fmt_str = f'Dataset {self.__class__.__name__}\n'
        fmt_str += f'    Number of datapoints: {self.__len__()}\n'
        fmt_str += f'    Split: {"train" if self.train else "test"}\n'
        fmt_str += f'    Root Location: {self.root}\n'
        if self.transform is not identity:
            tmp = '    Transforms: '
            fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        if self.target_transform is not identity:
            tmp = '    Target Transforms: '
            fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class Slides(SemanticSegmentationDataset):
    def __init__(self, *args, **kwargs):
        if "images_dir" in  kwargs.keys():
            self.images_dir = kwargs["images_dir"]
        else:
            self.images_dir = "TrainingSlidesInstances"

        #**************************************************
        # convert to parameter train and test size
        # convert to parameter height and width
        #**************************************************
        #read initial values from segmentation.ini            
        ini_file = Path().absolute().parent / self.images_dir / "segmentation.ini"
        if ini_file.exists():
            print("reading values from ini file")
            seg_config = configparser.ConfigParser()
            seg_config.read(ini_file)
            # sizes of training and testing datasets
            self.train_size = int(seg_config['DEFAULT']["trainsize"])
            self.test_size  = int(seg_config['DEFAULT']["testsize"])
            # image dimensions
            self.height = int(seg_config['DEFAULT']["imgheight"])
            self.width = int(seg_config['DEFAULT']["imgwidth"])
        else:
            # sizes of training and testing datasets
            self.train_size = 16 
            self.test_size  = 4
            # sizes of training and testing datasets
            self.height = 300
            self.width = 800
            
        super(Slides, self).__init__(*args, **kwargs)
        self.class_to_idx, self.colours = self.read_label_file(self.processed_folder / 'label_colors.txt')
        with h5py.File(self.datafile, 'r') as f:
            counts = np.bincount(f['labels'][()].flatten(), minlength=len(self.class_to_idx))
            self.weights = torch.Tensor((1 / counts) / (1 / counts).sum())
        print(self.weights)

    def process_label_image_files(self, folder_path, colours, f_train, f_test):
        train_set = f_train.create_dataset('labels', (self.train_size, self.height, self.width), dtype=np.int64)
        test_set = f_test.create_dataset('labels', (self.test_size, self.height, self.width), dtype=np.int64)
        images = (Image.open(filename) for filename in sorted(folder_path.glob('*.png')))

        values = ((np.array(colours) // 255) * np.array([1, 2, 4]).reshape(1, 3)).sum(axis=1)
        key = np.argsort(values)
        values.sort()

        for i, image in enumerate(images):
            image = 1 * (np.asarray(image) > 128)
            image_colours = (image * np.array([1, 2, 4]).reshape(1, 1, 3)).sum(axis=2)
            index = np.digitize(image_colours.ravel(), values, right=True).reshape(self.height, self.width)

            if i < self.train_size:
                train_set[i] = key[index]
                #test_set[i] = key[index]  # hack!!
            else:
                test_set[i - self.train_size] = key[index]

    def process_instance_image_files(self, folder_path, f_train, f_test):
        train_set = f_train.create_dataset('instances', (self.train_size, self.height, self.width), dtype=np.int64)
        test_set = f_test.create_dataset('instances', (self.test_size, self.height, self.width), dtype=np.int64)
        images = (Image.open(filename) for filename in sorted(folder_path.glob('*.png')))

        for i, image in enumerate(images):
            image = np.asarray(image)
            image_colors = image.reshape(-1, 3)
            colors, indices = np.unique(image_colors, axis=0, return_inverse=True)

            if i < self.train_size:
                train_set[i] = indices.reshape(image.shape[:2])
                #test_set[i] = indices.reshape(image.shape[:2])  # hackl!!
            else:
                test_set[i - self.train_size] = indices.reshape(image.shape[:2])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, labels, instances)
        """
        with h5py.File(self.datafile, 'r') as f:
            img = f['images'][index]
            labels = f['labels'][index]
            instances = f['instances'][index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        labels = Image.fromarray(labels.astype(np.uint8), mode='L')
        instances = Image.fromarray(instances.astype(np.uint8), mode='L')

        seed = np.random.randint(2147483647)
        random.seed(seed)
        img = self.transform(img)
        random.seed(seed)
        labels = self.target_transform(labels)
        random.seed(seed)
        instances = self.target_transform(instances)

        return img, labels, instances

    def download(self):
        self.raw_folder.mkdir(exist_ok=True, parents=True)
        self.processed_folder.mkdir(exist_ok=True, parents=True)

        folder = Path().absolute().parent / self.images_dir

        print(f'Copying images')

        (self.raw_folder / 'images').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.JPG')):
            shutil.copy(filename, self.raw_folder / 'images' / filename.name)

        print(f'Copying labels')

        (self.raw_folder / 'labels').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.png')):
            if 'label' in str(filename):
                shutil.copy(filename, self.raw_folder / 'labels' / filename.name)

        print(f'Copying instances')

        (self.raw_folder / 'instances').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.png')):
            if 'instance' in str(filename):
                shutil.copy(filename, self.raw_folder / 'instances' / filename.name)

        print(f'Copying class file')

        shutil.copy(folder / 'label_colours.txt', self.processed_folder / 'label_colors.txt')

        # process and save as torch files
        print('Processing...')

        self.class_to_idx, self.colours = self.read_label_file(self.processed_folder / 'label_colors.txt')

        with h5py.File(self.training_file, 'w') as f_train, h5py.File(self.test_file, 'w') as f_test:
            self.process_raw_image_files(self.raw_folder / 'images', f_train, f_test, extension='*.JPG')
            self.process_label_image_files(self.raw_folder / 'labels', self.colours, f_train, f_test)
            self.process_instance_image_files(self.raw_folder / 'instances', f_train, f_test)
        print('Done!')

class HerbariumSheets(SemanticSegmentationDataset):
    def __init__(self, *args, **kwargs):
        if "images_dir" in  kwargs.keys():
            self.images_dir = kwargs["images_dir"]
        else:
            self.images_dir = "TrainingHerbariumSheets"
        #**************************************************
        # convert to parameter train and test size
        # convert to parameter height and width
        #**************************************************
        #read initial values from segmentation.ini            
        ini_file = Path().absolute().parent / self.images_dir / "segmentation.ini"
        if ini_file.exists():
            print("reading values from ini file")
            seg_config = configparser.ConfigParser()
            seg_config.read(ini_file)
            # sizes of training and testing datasets
            self.train_size = int(seg_config['DEFAULT']["trainsize"])
            self.test_size  = int(seg_config['DEFAULT']["testsize"])
            # image dimensions
            self.height = int(seg_config['DEFAULT']["imgheight"])
            self.width = int(seg_config['DEFAULT']["imgwidth"])
        else:
            # sizes of training and testing datasets
            self.train_size = 200 
            self.test_size  = 50
            # sizes of training and testing datasets
            self.height = 1764
            self.width = 1169
                
        super(HerbariumSheets, self).__init__(*args, **kwargs)
        # read color classes from the file and asign them to:
        # self.class_to_idx as labels
        # self.colours as color codes
        self.class_to_idx, self.colours = self.read_label_file(self.processed_folder / 'label_colors.txt')
        with h5py.File(self.datafile, 'r') as f:
            counts = np.bincount(f['labels'][()].flatten(), minlength=len(self.class_to_idx))
            self.weights = torch.Tensor((1 / counts) / (1 / counts).sum())
            

    def process_label_image_files(self, folder_path, colours, f_train, f_test):
        train_set = f_train.create_dataset('labels', (self.train_size, self.height, self.width), dtype=np.int64)
        test_set = f_test.create_dataset('labels', (self.test_size, self.height, self.width), dtype=np.int64)
        images = (Image.open(filename) for filename in sorted(folder_path.glob('*.png')))

        values = ((np.array(colours) // 255) * np.array([1, 2, 4]).reshape(1, 3)).sum(axis=1)
        key = np.argsort(values)
        values.sort()
        j=k=l=0
        for i, image in enumerate(images):
            image = 1 * (np.asarray(image) > 128)
            image_colours = (image * np.array([1, 2, 4]).reshape(1, 1, 3)).sum(axis=2)
            index = np.digitize(image_colours.ravel(), values, right=True).reshape(self.height, self.width)
        # split get 2 of every ten into the test set
            if j>7:
                #add image to the test set
                test_set[k] = key[index]
                k+=1
            else:
                #add image to the train set
                train_set[l] = key[index]
                l+=1
            j+=1
            if j>9:
                j=0

    def process_instance_image_files(self, folder_path, f_train, f_test):
        train_set = f_train.create_dataset('instances', (self.train_size, self.height, self.width), dtype=np.int64)
        test_set = f_test.create_dataset('instances', (self.test_size, self.height, self.width), dtype=np.int64)
        images = (Image.open(filename) for filename in sorted(folder_path.glob('*.png')))
        j=k=l=0
        for i, image in enumerate(images):
            image = np.asarray(image)
            image_colors = image.reshape(-1, 3)
            colors, indices = np.unique(image_colors, axis=0, return_inverse=True)
        # split get 2 of every ten into the test set
            if j>7:
                #add image to the test set
                test_set[k] = indices.reshape(image.shape[:2])
                k+=1
            else:
                #add image to the train set
                train_set[l] = indices.reshape(image.shape[:2])
                l+=1
            j+=1
            if j>9:
                j=0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, labels, instances)
        """
        with h5py.File(self.datafile, 'r') as f:
            img = f['images'][index]
            labels = f['labels'][index]
            instances = f['instances'][index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        labels = Image.fromarray(labels.astype(np.uint8), mode='L')
        instances = Image.fromarray(instances.astype(np.uint8), mode='L')

        seed = np.random.randint(2147483647)
        random.seed(seed)
        img = self.transform(img)
        random.seed(seed)
        labels = self.target_transform(labels)
        random.seed(seed)
        instances = self.target_transform(instances)

        return img, labels, instances

    def download(self):
        #**************************************************
        # convert to parameter directory structure
        #**************************************************

        self.raw_folder.mkdir(exist_ok=True, parents=True)
        self.processed_folder.mkdir(exist_ok=True, parents=True)

        folder = Path().absolute().parent / self.images_dir

        print(f'Copying images')

        (self.raw_folder / 'images').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.JPG')):
            shutil.copy(filename, self.raw_folder / 'images' / filename.name)

        print(f'Copying labels')

        (self.raw_folder / 'labels').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*_labels.png')):
            if 'label' in str(filename):
                shutil.copy(filename, self.raw_folder / 'labels' / filename.name)

        print(f'Copying instances')

        (self.raw_folder / 'instances').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*_instances.png')):
            if 'instance' in str(filename):
                shutil.copy(filename, self.raw_folder / 'instances' / filename.name)

        print(f'Copying class file')
        
        #******************************************************
        # convert to parameter location and name of label file
        #******************************************************

        shutil.copy(folder / 'label_colours.txt', self.processed_folder / 'label_colors.txt')

        # process and save as torch files
        print('Processing...')
        
        self.class_to_idx, self.colours = self.read_label_file(self.processed_folder / 'label_colors.txt')

        with h5py.File(self.training_file, 'w') as f_train, h5py.File(self.test_file, 'w') as f_test:
            self.process_raw_image_files(self.raw_folder / 'images', f_train, f_test, extension='*.JPG')
            self.process_label_image_files(self.raw_folder / 'labels', self.colours, f_train, f_test)
            self.process_instance_image_files(self.raw_folder / 'instances', f_train, f_test)
        print('Done!')


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=identity):
        self.samples = list(Path(root).glob('*.JPG'))

        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path)

        return self.transform(img)

    def __len__(self):
        return len(self.samples)


class SemiSupervisedDataLoader:
    def __init__(self, loader_labelled, loader_unlabelled):
        self.labelled = loader_labelled
        self.unlabelled = loader_unlabelled

    def __iter__(self):
        return chain(*zip(self.unlabelled, cycle(self.labelled)))
