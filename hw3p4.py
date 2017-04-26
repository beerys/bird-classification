# Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13

import numpy as np
from get_data_info import get_data_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
#from keras import backend as K
#from keras.models import load_model
#from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.applications.resnet50 import ResNet50
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
test_folder = 'Test_Warp'
train_folder = 'Train_Warp'
filename = 'Bird_Model_3'
filepath = filename + '.h5'

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

#create callbacks
callbacks = [ModelCheckpoint('models/Bird_Model_1-{epoch:02d}-{val_acc:.4f}.hdf5'),CSVLogger('Bird_Model_1-history', separator=',', append=False)]

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator()

#fit the model (should I specify classes?  How do I split the training and test data)
history = model.fit_generator(datagen.flow_from_directory(directory=train_folder, target_size=(256,256),classes=classes),
					validation_data=datagen.flow_from_directory(directory=test_folder, target_size=(256,256),classes=classes),
					validation_steps=len(x_test_names)/batch_size,
                    epochs=epochs,
                    steps_per_epoch=len(x_train_names)/batch_size,
                    callbacks=callbacks,
                    verbose=1)

model.save(filepath)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'], loc = 'upper left')
plt.savefig(filename + '_accuracy.png', bbox_inches='tight')

#confusion matrix



