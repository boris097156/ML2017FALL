import csv
import numpy as np

EMOTION_CLASS = 7
IMG_ROW = 48
IMG_COL = 48
CHANNEL = 1

def read_csv(file_name, mode):
	x = [];
	y = [];
	with open(file_name, 'r', encoding='ISO-8859-1') as csv_train:
		reader = csv.reader(csv_train, delimiter=",")
		next(reader)
		for row in reader:
			if mode == "train":
				tmp = [0 for _ in range(EMOTION_CLASS)]
				tmp[int(row[0])] = 1
				y.append(tmp)
			r = row[1].strip().split()
			r = list(map(float, r))
			x.append(r)
	x = np.array(x, dtype=np.float32)
	y = np.array(y)
	return x.reshape(x.shape[0], IMG_ROW, IMG_COL, CHANNEL), y

def data_normalize(data):
	mean = np.mean(data, axis=0)
	data -= mean
	data /= 128