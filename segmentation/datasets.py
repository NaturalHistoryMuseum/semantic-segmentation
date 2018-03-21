from pathlib import Path
import random
import shutil
import tempfile
import urllib.request
import zipfile

import h5py
import numpy as np
from PIL import Image
from torch.utils import data


def identity(x):
    """To be used in place of explicitly checking whether there are transforms to be applied"""
    return x


class SemanticSegmentationDataset(data.Dataset):
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

    def __init__(self, root, train=True, transform=identity, target_transform=identity, download=False):
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

    def process_raw_image_files(self, folder_path, f_train, f_test, extension='*.png'):
        train_set = f_train.create_dataset('images', (self.train_size, self.height, self.width, 3), dtype=np.float32)
        test_set = f_test.create_dataset('images', (self.test_size, self.height, self.width, 3), dtype=np.float32)
        images = (Image.open(filename) for filename in sorted(folder_path.glob(extension)))
        for i, image in enumerate(images):
            if i < self.train_size:
                train_set[i] = np.asarray(image) / 255
                test_set[i] = np.asarray(image) / 255  # hack!!
            else:
                test_set[i - self.train_size] = np.asarray(image) / 255

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


class CamVid(SemanticSegmentationDataset):
    """`CamVid <http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = {'raw': 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip',
            'labels': 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip',
            'classes': 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/label_colors.txt'}

    def __init__(self, *args, **kwargs):
        super(CamVid, self).__init__(*args, **kwargs)
        self.train_size = 525
        self.test_size = 176
        self.height = 720
        self.width = 960

    def process_label_image_files(self, folder_path, colours, f_train, f_test):
        train_set = f_train.create_dataset('labels', (self.train_size, self.height, self.width), dtype=np.int64)
        test_set = f_test.create_dataset('labels', (self.test_size, self.height, self.width), dtype=np.int64)
        images = (Image.open(filename) for filename in sorted(folder_path.glob('*.png')))

        values = ((np.array(colours) // 64) * np.array([1, 4, 16]).reshape(1, 3)).sum(axis=1)
        key = np.argsort(values)
        values.sort()

        for i, image in enumerate(images):
            image_colours = ((np.asarray(image) // 64) * np.array([1, 4, 16]).reshape(1, 1, 3)).sum(axis=2)
            index = np.digitize(image_colours.ravel(), values, right=True).reshape(self.height, self.width)

            if i < self.train_size:
                train_set[i] = key[index]
            else:
                test_set[i - self.train_size] = key[index]

    def download(self):
        """Download the CamVid data if it doesn't exist in processed_folder already."""
        self.raw_folder.mkdir(exist_ok=True, parents=True)
        self.processed_folder.mkdir(exist_ok=True, parents=True)

        print(f'Downloading {self.urls["raw"]}')

        data = urllib.request.urlopen(self.urls["raw"])
        with tempfile.NamedTemporaryFile('w') as tmp:
            tmp.write(data.read())
            with zipfile.ZipFile(tmp.name) as zip_f:
                zip_f.extractall(self.raw_folder)

        print(f'Downloading {self.urls["labels"]}')

        data = urllib.request.urlopen(self.urls["labels"])
        with tempfile.NamedTemporaryFile('wb') as tmp:
            tmp.write(data.read())
            with zipfile.ZipFile(tmp.name) as zip_f:
                zip_f.extractall(self.raw_folder / 'LabeledApproved_full')

        print(f'Downloading {self.urls["classes"]}')

        data = urllib.request.urlopen(self.urls["classes"])
        with open(self.processed_folder / 'label_colors.txt', 'wb') as class_list:
            class_list.write(data.read())

        # process and save as torch files
        print('Processing...')

        self.class_to_idx, colours = self.read_label_file(self.processed_folder / 'label_colors.txt')

        with h5py.File(self.training_file, 'w') as f_train, h5py.File(self.test_file, 'w') as f_test:
            self.process_raw_image_files(self.raw_folder / '701_StillsRaw_full', f_train, f_test)
            self.process_label_image_files(self.raw_folder / 'LabeledApproved_full', colours, f_train, f_test)

        print('Done!')


class Slides(SemanticSegmentationDataset):
    def __init__(self, *args, **kwargs):
        self.train_size = 13
        self.test_size = 13
        self.height = 330
        self.width = 800
        super(Slides, self).__init__(*args, **kwargs)
        self.class_to_idx, self.colours = self.read_label_file(self.processed_folder / 'label_colors.txt')

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
                test_set[i] = key[index]  # hack!!
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
                test_set[i] = indices.reshape(image.shape[:2])  # hackl!!
            else:
                test_set[i - self.train_size] = indices.reshape(image.shape[:2])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
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

        folder = Path().absolute().parent / 'TrainingSlidesInstances'

        print(f'Copying images')

        (self.raw_folder / 'images').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.JPG')):
            shutil.copy(filename, self.raw_folder / 'images' / filename.parts[-1])

        print(f'Copying labels')

        (self.raw_folder / 'labels').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.png')):
            if 'label' in str(filename):
                shutil.copy(filename, self.raw_folder / 'labels' / filename.parts[-1])

        print(f'Copying labels')

        (self.raw_folder / 'instances').mkdir(exist_ok=True)
        for filename in sorted(folder.glob('*.png')):
            if 'instance' in str(filename):
                shutil.copy(filename, self.raw_folder / 'instances' / filename.parts[-1])

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
