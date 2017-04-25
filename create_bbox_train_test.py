#Sara Beery


import os
from shutil import copyfile
#from matplotlib import image
from PIL import Image

source_dir = 'CUB_200_2011/CUB_200_2011/'
source_img_dir = source_dir + 'images/'
train_test_split = [line.rstrip().split() for line in open(source_dir + 'train_test_split.txt')]
image_paths = [line.rstrip().split() for line in open(source_dir + 'images.txt')]
bboxes = [line.rstrip().split() for line in open(source_dir + 'bounding_boxes.txt')]

num_test = 0
num_train = 0
for i in range(len(train_test_split)):
    img_dir = image_paths[i][1].split('/')[0]
    #bbox = [bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]]
    if int(train_test_split[i][1]):
        new_dir = 'Train_Bbox/'
        num_train += 1
    else:
        new_dir = 'Test_Bbox/'
        num_test += 1

    if not os.path.exists(new_dir + img_dir):
        os.makedirs(new_dir + img_dir)

    #open and crop images
    #img = image.imread(source_img_dir + image_paths[i][1])
    img = Image.open(source_img_dir + image_paths[i][1])
    box = (int(float(bboxes[i][1])), int(float(bboxes[i][2])), int(float(bboxes[i][3])+float(bboxes[i][1])), int(float(bboxes[i][4])+float(bboxes[i][2])))
    img_cropped = img.crop(box)
    img_cropped.save(new_dir + image_paths[i][1])
    #copyfile(img_cropped, new_dir + image_paths[i][1])

print(num_train)
print(num_test)




