from pathlib import Path
from shutil import copyfile
import random

list_full = []
for filename in Path('data', 'nhm_resized').iterdir():
    list_full.append(filename.name)
print("Slides total: ", len(list_full))

list_validate = []
for filename in Path('data', 'nhm_validate').iterdir():
    nameonly = filename.name 
    if "labels" not in nameonly and "instances" not in nameonly:
        list_validate.append(filename.name)

print("Segmented slides: ",len(list_validate))

list_test = []
for filename in Path('data', 'nhm_test').iterdir():
    nameonly = filename.name 
    if "labels" not in nameonly and "instances" not in nameonly:
        list_test.append(filename.name)

print("Testing slides: ",len(list_test))

list_not_used = list(set(list_full) ^ set(list_validate) ^ set(list_test))

print("Not used slides:", len(list_not_used))

#random list of unused slides to build new training set
rand_smpl = [ list_not_used[i] for i in sorted(random.sample(range(len(list_not_used)), 20)) ]

print(rand_smpl)


##for cpy_file in rand_smpl:
##    print("copy:",  str(Path('data', 'nhm_resized',cpy_file)), "to:", str(Path('data', 'nhm_train',cpy_file)))
##    copyfile(str(Path('data', 'nhm_resized',cpy_file)),
##             str(Path('data', 'nhm_train',cpy_file)))

    
