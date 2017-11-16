import csv
import sys
import random
import numpy as np
import my_function as func
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import l2
from keras.layers.merge import add, concatenate
from keras.utils import plot_model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

np.set_printoptions(threshold=np.nan)

train_datagen = ImageDataGenerator(vertical_flip=False, horizontal_flip=True, height_shift_range=0.1, width_shift_range=0.1, rotation_range=20.0)

conv = np.array([32, 64, 64, 128, 128])
dense = np.array([128, 128, 128])
drop = np.array([0.5, 0.5, 0.5])

#LOG_NUM = sys.argv[1]
TRAIN_FILE = sys.argv[1]
#DIR = LOG_NUM

BATCH_SIZE = 32
EPOCHS = 65
EMOTION_CLASS = 7
IMAGE_ROW = 48
IMAGE_COL = 48
CHANNEL = 1
MODEL_NAME =  'model0.h5'
#CHECK_NAME = DIR + '/model_check.h5'

train_x, train_y = func.read_csv(TRAIN_FILE, 'train')
func.data_normalize(train_x)
VALID = train_x.shape[0]
valid_x = np.copy(train_x[VALID:])
valid_y = np.copy(train_y[VALID:])
train_x = np.copy(train_x[:VALID])
train_y = np.copy(train_y[:VALID])
train_datagen.fit(train_x, seed=7)
#valid_datagen.fit(valid_x, seed=17)


def add_shortcut(input, residual):
	input_shape = K.int_shape(input)
	residual_shape = K.int_shape(residual)
	stride_width = int(round(input_shape[1] / residual_shape[1]))
	stride_height = int(round(input_shape[2] / residual_shape[2]))
	equal_channels = input_shape[3] == residual_shape[3]
	shortcut = input
	# 1 X 1 conv if shape is different. Else identity.
	if stride_width > 1 or stride_height > 1 or not equal_channels:
		shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
							strides=(stride_width, stride_height),
							padding="valid",
							kernel_initializer="he_normal",
							kernel_regularizer=l2(0.001))(input)

	return add([shortcut, residual])

input = Input(shape=(IMAGE_ROW, IMAGE_COL, CHANNEL))
conv0 = (Conv2D(conv[0], (3, 3), activation='relu', padding='same'))(input)	#46
norm0 = BatchNormalization()(conv0)
pool0 = MaxPooling2D(pool_size=(2, 2))(norm0)
#drop0 = Dropout(drop[0])(pool0)

conv1 = Conv2D(conv[1], (3, 3), activation='relu', padding='same')(pool0)	#46
norm1 = BatchNormalization()(conv1)
conv2 = Conv2D(conv[2], (3, 3), activation='relu', padding='same')(norm1)	#46
norm2 = BatchNormalization()(conv2)
pool1 = MaxPooling2D(pool_size=(2, 2))(norm2)
drop1 = Dropout(drop[0])(pool1)

residual0 = add_shortcut(pool0, drop1)

conv3 = Conv2D(conv[3], (3, 3), activation='relu', padding='same')(residual0)	#46
norm3 = BatchNormalization()(conv3)
conv4 = Conv2D(conv[4], (3, 3), activation='relu', padding='same')(norm3)	#46
norm4 = BatchNormalization()(conv4)
pool2 = MaxPooling2D(pool_size=(2, 2))(norm4)
drop2 = Dropout(drop[1])(pool2)

residual1 = add_shortcut(drop1, drop2)

flatten = Flatten()(residual1)

dense0 = Dense(units=dense[0], activation='relu')(flatten)
norm5 = BatchNormalization()(dense0)
dense1 = Dense(units=dense[1], activation='relu')(norm5)
norm6 = BatchNormalization()(dense1)
dense2 = Dense(units=dense[2], activation='relu', kernel_regularizer=l2(0.1))(norm6)
norm7 = BatchNormalization()(dense2)
dense3 = Dense(units=EMOTION_CLASS, activation='softmax')(norm7)

model = Model(inputs=input, outputs=dense3)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
#earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
#checkpoint = ModelCheckpoint(CHECK_NAME, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#model.fit(train_x, train_y, batch_size=BATCH_SIZE, callbacks=[earlyStopping, checkpoint], shuffle=True, verbose=1, epochs=EPOCHS, validation_split=0.15)

model.fit_generator(train_datagen.flow(train_x, train_y, batch_size=BATCH_SIZE), 
						steps_per_epoch=train_x.shape[0]/BATCH_SIZE, 
						validation_data=(valid_x, valid_y),
						epochs=EPOCHS, 
						verbose=1)

model.save(MODEL_NAME)
#plot_model(model, to_file='model.png')
