import csv
import numpy as np
import my_fs as fs

''' TO-DO list
validation sets
stochastic
divide into train & test parts
more features
more power(次方)
'''

#Basic variables
ITEM_NUM = 18
DATA_NUM = 5760
LOG_NAME = 'log/log.txt'
WLOG_NAME = 'weight.out'
FWLOG_NAME = 'fs_weight.out'

VALID_NUM = 600
FEATURE_NUM = 163
INIT_LR = 0.5
INIT_W = 0.5

def log_init():
	with open(LOG_NAME, 'w') as log:
		log.write('train/valid: ' + str((DATA_NUM-VALID_NUM)) + '/'  + str(VALID_NUM) + '\n')
		log.write('learn_rate:' + str(INIT_LR) + '\n')
		log.write('features:' + str(FEATURE_NUM) + '\n')

def read_csv(file_name, my_data, start, end):
	with open(file_name, 'r', encoding='ISO-8859-1') as csv_file:
		raw_file = csv.reader(csv_file , delimiter=",")
		#discard 1st line of train data
		if start == 3:
			next(raw_file)
		n_row = 0;
		for row in raw_file:
			for item in range(start,end):
				if row[item] != "NR":
					my_data[n_row%ITEM_NUM].append( float( row[item] ) )
				else:
					my_data[n_row%ITEM_NUM].append( float( 0 ) )	
			n_row += 1

def reshape_train(train_data, data_x, data_y):
	for month in range(12):
		for hour_start in range(471):
			data_x.append([1])
			for item in range(ITEM_NUM):
				for hour in range(9):
					data_x[471*month+hour_start].append(train_data[item][480*month+hour_start+hour])
			data_y.append(train_data[9][480*month+hour_start+9])

def train(train_x, train_y, weight):
	prev_gra = np.zeros((FEATURE_NUM,), dtype=np.float32)
	learn_rate = INIT_LR
	laps=0

	with open(LOG_NAME, 'a') as log:
		while(1):
			laps += 1
			my_y = np.dot(train_x, weight)
			dif = my_y-train_y
			avg_lost = np.sum(np.square(dif))/DATA_NUM
			#logging
			log.write(str(laps) + '\t' + str(avg_lost) + '\n') 
			if abs(avg_lost) <= 0.1161 or laps >= 9000:
				print('laps: ' + str(laps) + '\t' + 'avg_lost: ' + str(avg_lost))
				return
			gra = np.dot(train_x.transpose(), dif)
			prev_gra += gra**2
			ada = np.sqrt(prev_gra)
			weight -= learn_rate*gra/ada

def main():
	log_init()

	#read in train csv
	train_data = [[] for _ in range(ITEM_NUM)]
	read_csv('train.csv', train_data, 3, 27)

	#feature scaling for both data
	feature_wegith = [[] for _ in range(ITEM_NUM)]
	fs_weight = fs.cal_weight(train_data)
	fs.scale_down(train_data, fs_weight)
	np.savetxt(FWLOG_NAME, fs_weight, delimiter=',')

	#reshape and divide into x and y
	data_x = []
	data_y = []
	reshape_train(train_data, data_x, data_y)

	data_x = np.array(data_x)
	data_y = np.array(data_y)
	
	#train & valid	
	weight = np.full((FEATURE_NUM,), INIT_W, dtype=np.float32)
	train(data_x, data_y, weight)
	#valid()

	np.savetxt(WLOG_NAME, weight, delimiter=',')

if __name__ == "__main__":
    main()
