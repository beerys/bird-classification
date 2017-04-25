#Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13


import numpy as np
import matplotlib.pyplot as plt

import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.models import Model
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import ModelCheckpoint

def get_data_info(num_ims):
    source_dir = 'CUB_200_2011/CUB_200_2011/'
    class_file = source_dir + 'image_class_labels.txt'
    image_file = source_dir + 'images.txt'
    split_file = source_dir + 'train_test_split.txt'
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    classes = []
    f1 = open(class_file, 'r')
    f2 = open(image_file, 'r')
    f3 = open(split_file, 'r')
    count = 0
    while count < num_ims:
        line = f3.readline().split()
        file_name = f2.readline().split()[1]
        class_name = file_name.split('/')[0]
        class_num = float(f1.readline().split()[1])-1
        if class_name not in classes:
            classes.append(class_name)
        if float(line[1]):
            x_train.append(file_name)
            y_train.append(class_num)
        else:
            x_test.append(file_name)
            y_test.append(class_num)
        count += 1
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return x_train, x_test, y_train, y_test, classes


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
test_folder = 'Test'
train_folder = 'Train'
filepath = "Bird_Model_1.h5"

x_train_names, x_test_names, y_train, y_test, classes = get_data_info(num_ims)

model = load_model(filepath)

datagen = ImageDataGenerator()
test_generator = datagen.flow_from_directory(directory=test_folder,
                                             target_size=(256,256),
                                             batch_size = 1,
                                             classes=classes,
                                             shuffle = False)

count = 0
y_pred = []
while count < len(x_test_names):
    count += 1
    (x,y) = test_generator.next()
    y_pred.append(model.predict_classes(x, batch_size=1, verbose=0))
cnf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(filepath + '_confmat.png', bbox_inches='tight')