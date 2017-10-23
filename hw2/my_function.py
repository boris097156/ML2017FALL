import csv
import numpy as np

#Change variables here
CARE_ITEM = 106
FEATURE_NUM = 1 + CARE_ITEM

LOG_NAME = 'log/log'
WLOG_NAME = 'weight.out'
FWLOG_NAME = 'fs_weight.out'

def read_csv(file_name):
	data = []
	with open(file_name, 'r', encoding='ISO-8859-1') as csv_file:
		raw_file = csv.reader(csv_file, delimiter=",")
		next(raw_file)
		for row in raw_file:
			tmp = []
			for item in row:
				tmp.append(float(item))
			if len(tmp) > 1:
				data.append(tmp)
			else:
				data.append(tmp[0])
	return np.array(data, dtype=np.float64)	

def cal_weight(train_data):
	train_size = train_data.shape[1]
	my_sum = np.sum(train_data, axis=1, keepdims=True)
	my_avg = my_sum/(train_size)
	my_squ = np.sum((train_data-my_avg)**2, axis=1, keepdims=True)
	my_sd = (my_squ/train_size)**0.5
	fs_weight = np.concatenate((my_avg, my_sd), axis=1)
	return fs_weight

def scale_down(my_data, fs_weight):
	data_size = my_data.shape[1]
	for i in range(CARE_ITEM):
		my_data[i] -= fs_weight[i][0]
		my_data[i] /= fs_weight[i][1]
	return

def sigmoid(z):
	res = float(1)/(1.0 + np.exp(-z))
	return np.clip(res, 1e-8, (1-1e-8))

def csv_result(my_y, OUTPUT_NAME):
    with open(OUTPUT_NAME, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        writer.writerow(['id']+['label'])
        for i in range(my_y.shape[0]):
            writer.writerow([str(i+1)]+[str(int(my_y[i]))])
