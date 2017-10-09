import csv
import numpy as np

#Basic variables
ITEM_NUM = 18
LOG_NUM = 1

#Change variables here
CARE_HOUR = 9
CARE_ITEM = 10
PM_POSITION = 2
DIR = 'O3_PM10_PM2.5_combination-7500'

FEATURE_NUM = 1 + CARE_HOUR*CARE_ITEM
LOG_NAME = DIR + '/log/log'
WLOG_NAME = DIR + '/weight.out'
FWLOG_NAME = DIR + '/fs_weight.out'

def prune_data(origin_data):
	my_data = []
	my_data.append(origin_data[7]) #O3
	my_data.append(origin_data[8]) #PM10
	my_data.append(origin_data[9]) #PM2.5
	my_data.append(np.square(origin_data[7])) #O3 ^2
	my_data.append(np.square(origin_data[8])) #PM10 ^2
	my_data.append(np.square(origin_data[9])) #PM2.5 ^2
	my_data.append(origin_data[7]*origin_data[8]) #O3 * PM10
	my_data.append(origin_data[8]*origin_data[9]) #PM10 * PM2.5
	my_data.append(origin_data[9]*origin_data[7]) #PM2.5 * O3
	my_data.append(origin_data[7]*origin_data[8]*origin_data[9]) #O3 * PM10 * PM2.5
	return np.array(my_data, dtype=np.float64)	

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
	return prune_data(np.array(origin_data))

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
	my_data *= fs_weight[PM_POSITION][1]
	my_data += fs_weight[PM_POSITION][0]
	return

def cal_y(my_x, weight, fs_weight):
	my_y = np.dot(my_x, weight)
	scale_up(my_y, fs_weight)
	return my_y
