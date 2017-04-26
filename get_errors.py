#Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from get_data_info import get_data_info

import keras
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator


batch_size = 32
train_all_classes = True
create_train_test_dirs = True
layers_to_train = 1
img_rows = 150
img_cols = 150
epochs = 10

if train_all_classes:
    num_ims = 11788
    num_classes = 200
else:
    num_ims = 1115
    num_classes = 20

image_folder = 'CUB_200_2011/CUB_200_2011/images'
image_folder1 = 'Test_Bbox'
test_folder = 'Test_Warp'
train_folder = 'Train_Warp'
filename1 = 'Bird_Model_2'
filepath1 = filename1 + '.h5'
filename2 = 'Bird_Model_3'
filepath2 = filename2 + '.h5'

x_train_names, x_test_names, y_train, y_test, classes = get_data_info(num_ims)

model = load_model(filepath1)

datagen = ImageDataGenerator()
test_generator = datagen.flow_from_directory(directory=test_folder,
                                             target_size=(256,256),
                                             batch_size = 1,
                                             classes=classes,
                                             shuffle = False)

count = 0
y_pred = []

pred = model.predict_generator(test_generator, len(x_test_names), verbose=1)
y_pred = []
y_class = []
for i in range(len(test_generator.filenames)):
    y_pred += [np.argmax(pred[i])]
    y_class += [int(test_generator.filenames[i][:3])-1]

diffs1 = [i for i in range(len(y_pred)) if y_pred[i] != y_class[i]]

model = load_model(filepath2)

datagen = ImageDataGenerator()
test_generator = datagen.flow_from_directory(directory=test_folder,
                                             target_size=(256,256),
                                             batch_size = 1,
                                             classes=classes,
                                             shuffle = False)

count = 0
y_pred = []

pred = model.predict_generator(test_generator, len(x_test_names), verbose=1)
y_pred = []
y_class = []
for i in range(len(test_generator.filenames)):
    y_pred += [np.argmax(pred[i])]
    y_class += [int(test_generator.filenames[i][:3])-1]

diffs2 = [i for i in range(len(y_pred)) if y_pred[i] != y_class[i]]

files2 = [test_generator.filenames[i] for i in diffs2]

fixed = [i for i in diffs1 if i not in diffs2]

fixedFiles = [test_generator.filenames[i] for i in fixed]

#num_fixed = max(len(fixedFiles),5)
plt.figure()
count = 0
plotnum = 1
for im in fixedFiles:
    if count < 5:
        plt.subplot(5,2,plotnum)
        img = Image.open(image_folder1 +'/'+ im)
        plt.imshow(img)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plotnum += 1
        plt.subplot(5,2,plotnum)
        img = Image.open(test_folder +'/'+ im)
        plt.imshow(img)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plotnum += 1
        count +=1
plt.savefig('fixes.png', bbox_inches='tight')
