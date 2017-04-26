

import os
from shutil import copyfile
#from matplotlib import image
import numpy as np
from skimage import io
from skimage.transform import SimilarityTransform, warp

source_dir = 'CUB_200_2011/CUB_200_2011/'
source_img_dir = source_dir + 'images/'
train_test_split = [line.rstrip().split() for line in open(source_dir + 'train_test_split.txt')]
image_paths = [line.rstrip().split() for line in open(source_dir + 'images.txt')]
bboxes = [line.rstrip().split() for line in open(source_dir + 'bounding_boxes.txt')]
part_names = [line.rstrip().split() for line in open(source_dir + 'parts/parts.txt')]
part_locs = [line.rstrip().split() for line in open(source_dir + 'parts/part_locs.txt')]


#calculate part locations
# [beak, crown, left eye, right eye, throat]
idealLocs = [[250, 124],
             [124, 124],
             [124, 200],
             [124, 80]]

num_test = 0
num_train = 0
for i in range(len(train_test_split)):

    img_dir = image_paths[i][1].split('/')[0]
    #bbox = [bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]]
    if int(train_test_split[i][1]):
        new_dir = 'Train_Warp/'
        num_train += 1
    else:
        new_dir = 'Test_Warp/'
        num_test += 1

    if not os.path.exists(new_dir + img_dir):
        os.makedirs(new_dir + img_dir)

    #open and warp images
    start = i*15
    imPartLocs = []
    keep = False
    partsIwant = [2, 5, 7, 15]
    if part_locs[start+7-1] is 0: ##head turned left
        partsIwant[2] = 11
    ignore = False
    for j in partsIwant:
        if part_locs[start+j-1][4] is 0:
            ignore = True
        imPartLocs.append([int(float(part_locs[start+j-1][2])), int(float(part_locs[start+j-1][3]))])
    if not ignore:
        #calculate warp
        st = SimilarityTransform()
        st.estimate(np.asarray(idealLocs),np.asarray(imPartLocs))
        img = io.imread(source_img_dir + image_paths[i][1])
        img_warped = warp(img, st, output_shape=(256,256))
        io.imsave(new_dir + image_paths[i][1],img_warped)
    #copyfile(img_cropped, new_dir + image_paths[i][1])

print(num_train)
print(num_test)




