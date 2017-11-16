import csv
import sys
import random
import numpy as np
import my_function as func
import keras
import os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras import regularizers
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

np.set_printoptions(threshold=np.nan)
datagen = ImageDataGenerator(featurewise_center=False, 
							samplewise_center='False', 
							featurewise_std_normalization=False,
							samplewise_std_normalization=False, 
							zca_whitening=False, 
							rotation_range=0, 
							width_shift_range=0.1,
							height_shift_range=0.1, 
							horizontal_flip=True, 
							vertical_flip=False)

conv = np.array([32, 64, 64, 64, 128])
dense = np.array([128, 128, 128])
drop = np.array([0.35, 0.5, 0.5])

#LOG_NUM = sys.argv[1]
#data_path = os.environ.get("GRAPE_DATASET_DIR")
#TRAIN_FILE = os.path.join(data_path, "data/train.csv")
TRAIN_FILE = sys.argv[1]
#DIR = LOG_NUM

my_batch_size = 128
my_epochs = 200
seed = 7
EMOTION_CLASS = 7
IMAGE_ROW = 48
IMAGE_COL = 48
CHANNEL = 1
MODEL_NAME = 'model2.h5'

np.random.seed(seed)
train_x, train_y = func.read_csv(TRAIN_FILE, 'train')
func.data_normalize(train_x)
datagen.fit(train_x)


model = Sequential()

model.add(Conv2D(conv[0], (3, 3), activation='relu', padding='same', input_shape=(IMAGE_ROW, IMAGE_COL, CHANNEL)))	#46
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))	

model.add(Conv2D(conv[1], (3, 3), activation='relu'))																#21
model.add(BatchNormalization())
model.add(Conv2D(conv[2], (3, 3), activation='relu'))																#8
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))	
model.add(Dropout(drop[1]))

model.add(Conv2D(conv[3], (3, 3), activation='relu'))																#8
model.add(BatchNormalization())
model.add(Conv2D(conv[4], (3, 3), activation='relu'))																#8
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))																		#1
model.add(Dropout(drop[2]))

model.add(Flatten())

model.add(Dense(units=dense[0], activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=dense[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=dense[2], activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(BatchNormalization())
model.add(Dense(units=EMOTION_CLASS, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='acc', patience=6, verbose=0, mode='auto')

model.fit(train_x, train_y, batch_size=my_batch_size, callbacks=[earlyStopping], shuffle=True, verbose=1, epochs=my_epochs)
model.save(MODEL_NAME)
'''
TEST_FILE = os.path.join(data_path, "data/test.csv")

model = load_model(MODEL_NAME)
test_x, label = func.read_csv(TEST_FILE, "test")
func.data_normalize(test_x)

ans = model.predict(test_x)
ans = np.argmax(ans, axis=1)
ans_list = ans.tolist()

with open(DIR+'/submission.csv', 'w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
	writer.writerow(['id']+['label'])
	for i in range(len(ans_list)):
		writer.writerow([str(i)]+[str(ans_list[i])])

#plot_model(model, to_file=DIR + '/model.png')
'''