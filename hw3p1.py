# Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13

import numpy as np
#import confusion_mat as cm
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator


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
train_all_classes = False
create_train_test_dirs = True
layers_to_train = 1
img_rows = 150
img_cols = 150
epochs = 2

if train_all_classes:
    num_ims = 11788
    num_classes = 200
else:
    num_ims = 1115
    num_classes = 20


image_folder = 'CUB_200_2011/CUB_200_2011/images'
test_folder = 'Test'
train_folder = 'Train'

x_train_names, x_test_names, y_train, y_test, classes = get_data_info(num_ims)

# convert class vectors to binary class matrices, subtract 1 to get correct classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

#create checkpoints (after model has been compiled)
# filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

#create Tensorboard Logs
#remember to pass this to your model while fitting!! model.fit(...inputs and parameters..., callbacks=[tbCallBack])
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', callbacks=[tbCallBack])

datagen = ImageDataGenerator()

#fit the model (should I specify classes?  How do I split the training and test data)
model.fit_generator(datagen.flow_from_directory(directory=train_folder, target_size=(256,256),classes=classes),
                    epochs=epochs,
                    steps_per_epoch=len(x_train_names))

