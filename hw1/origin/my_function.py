import csv
import numpy as np

#Basic variables
ITEM_NUM = 18
LOG_NUM = 3
LOG_NAME = 'log/log'
WLOG_NAME = 'weight.out'
FWLOG_NAME = 'fs_weight.out'

CARE_HOUR = 9
CARE_ITEM = 18
CARE_ITEM_START = 0
if CARE_ITEM == 1:
	CARE_ITEM_START = 9
PM_POSITION = 9 - CARE_ITEM_START
FEATURE_NUM = 1 + CARE_HOUR*CARE_ITEM

def read_csv(file_name, start, end):
	origin_data = [[] for _ in range(ITEM_NUM)]
	with open(file_name, 'r', encoding='ISO-8859-1') as csv_file:
		raw_file = csv.reader(csv_file , delimiter=",")
		#discard 1st line of train data
		if start == 3:
			next(raw_file)
		n_row = 0;
		for row in raw_file:
			for item in range(start,end):
				if row[item] != "NR":
					origin_data[n_row%ITEM_NUM].append( float( row[item] ) )
				else:
					origin_data[n_row%ITEM_NUM].append( float( 0 ) )	
			n_row += 1
	my_data = []
	for i in range(CARE_ITEM):
		my_data.append(origin_data[CARE_ITEM_START + i])	
	return np.array(my_data, dtype=np.float64)

def cal_weight(train_data):
	fs_weight = [[] for _ in range(CARE_ITEM)]
	train_size = train_data.shape[1]
	for i in range(CARE_ITEM):
		my_sum = np.sum(train_data[i])
		my_avg = my_sum/(train_size)
		my_squ = np.sum((train_data[i]-my_avg)**2)
		my_sd = (my_squ/train_size)**0.5
		fs_weight[i].append(my_avg)
		fs_weight[i].append(my_sd)
	return np.array(fs_weight, dtype=np.float64)

def scale_down(my_data, fs_weight):
	data_size = my_data.shape[1]
	for i in range(CARE_ITEM):
		my_data[i] -= fs_weight[i][0]
		my_data[i] /= fs_weight[i][1]
	return

def scale_up(my_data, fs_weight):
	data_size = my_data.shape[0]
	for i in range(data_size):
		my_data[i] *= fs_weight[PM_POSITION][1]
		my_data[i] += fs_weight[PM_POSITION][0]
	return

def cal_RMSE(my_y, real_y):
	num = my_y.shape[0]
	RMSE = (np.sum((my_y-real_y)**2)/num)**0.5
	return RMSE

def cal_y(my_x, weight, fs_weight):
	my_y = np.dot(my_x, weight)
	scale_up(my_y, fs_weight)
	return my_y
