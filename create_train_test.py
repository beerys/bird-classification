#Sara Beery


import os
from shutil import copyfile

source_dir = 'CUB_200_2011/CUB_200_2011/'
source_img_dir = source_dir + 'images/'
train_test_split = [line.rstrip().split() for line in open(source_dir + 'train_test_split.txt')]
image_paths = [line.rstrip().split() for line in open(source_dir + 'images.txt')]

num_test = 0
num_train = 0
for i in range(len(train_test_split)):
    img_dir = image_paths[i][1].split('/')[0]
    if int(train_test_split[i][1]):
        new_dir = 'Train/'
        num_train += 1
    else:
        new_dir = 'Test/'
        num_test += 1

    if not os.path.exists(new_dir + img_dir):
        os.makedirs(new_dir + img_dir)
    copyfile(source_img_dir + image_paths[i][1], new_dir + image_paths[i][1])

print(num_train)
print(num_test)