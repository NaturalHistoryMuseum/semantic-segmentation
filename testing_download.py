
from pathlib import Path
import shutil

root = Path('data').expanduser()
raw_folder = root / 'raw'
processed_folder = root / 'processed'

print root
#def download():

##self.raw_folder.mkdir(exist_ok=True, parents=True)
##self.processed_folder.mkdir(exist_ok=True, parents=True)
##
##folder = Path().absolute().parent / 'TrainingSlidesInstances'
##
##print(f'Copying images')
##
##(self.raw_folder / 'images').mkdir(exist_ok=True)
##for filename in sorted(folder.glob('*.JPG')):
##    shutil.copy(filename, self.raw_folder / 'images' / filename.name)
##
##print(f'Copying labels')
##
##(self.raw_folder / 'labels').mkdir(exist_ok=True)
##for filename in sorted(folder.glob('*.png')):
##    if 'label' in str(filename):
##        shutil.copy(filename, self.raw_folder / 'labels' / filename.name)
##
##print(f'Copying instances')
##
##(self.raw_folder / 'instances').mkdir(exist_ok=True)
##for filename in sorted(folder.glob('*.png')):
##    if 'instance' in str(filename):
##        shutil.copy(filename, self.raw_folder / 'instances' / filename.name)
##
##print(f'Copying class file')
##
##shutil.copy(folder / 'label_colours.txt', self.processed_folder / 'label_colors.txt')
##
### process and save as torch files
##print('Processing...')
##
