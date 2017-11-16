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

#data_path = os.environ.get("GRAPE_DATASET_DIR")
#TEST_FILE = os.path.join(data_path, "best_models4/test.csv")
TEST_FILE = sys.argv[1]
SUBMISSION_FILE = sys.argv[2]
test_x, label = func.read_csv(TEST_FILE, "test")
func.data_normalize(test_x)

model_list = [0, 2]
all_ans_list = []
vote_ans_list = []

for i in model_list:
#        MODEL_NAME = "best" + str(i) + "/model.h5"
	MODEL_NAME = "model" + str(i) + ".h5"
	model = load_model(MODEL_NAME)
	ans = model.predict(test_x)
	ans_list = ans.tolist()
	all_ans_list.append(ans_list)

for i in range(len(all_ans_list[0])):
	vote = np.zeros(7)
	for j in range(len(all_ans_list)):
		vote += all_ans_list[j][i]
	vote_ans_list.append(np.argmax(vote))

with open(SUBMISSION_FILE, 'w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
	writer.writerow(['id']+['label'])
	for i in range(len(vote_ans_list)):
		writer.writerow([str(i)]+[str(vote_ans_list[i])])
