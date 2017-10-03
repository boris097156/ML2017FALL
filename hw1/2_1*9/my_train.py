import csv
import sys
import numpy as np
import my_function as my

''' TO-DO list
more features
more power(次方)
'''

#Paths
LOG_NAME = my.LOG_NAME
WLOG_NAME = my.WLOG_NAME
FWLOG_NAME = my.FWLOG_NAME
TRAIN_NAME = '../train.csv'

#feature variables
CARE_HOUR = my.CARE_HOUR
CARE_ITEM = my.CARE_ITEM
FEATURE_NUM = my.FEATURE_NUM

#training variables
DATA_NUM = 5760 - CARE_HOUR*12
VALID_NUM = int(sys.argv[1])
TRAIN_NUM = DATA_NUM - VALID_NUM
BATCH_SIZE = 100
INIT_LR = 0.5
INIT_W = 0.5
THR_LAP = 1000
THR_LOST = 0

def log_init(log_name):
	with open(log_name, 'w') as log:
		log.write('train/valid: ' + str(TRAIN_NUM) + '/'  + str(VALID_NUM) + '\n')
		log.write('features: ' + str(FEATURE_NUM) + '\n')
		log.write('care hour: ' + str(CARE_HOUR) + '\n')
		log.write('care item: ' + str(CARE_ITEM) + '\n')
		log.write('learn_rate: ' + str(INIT_LR) + '\n')
		log.write('init_weight: ' + str(INIT_W) + '\n')

def reshape_train(train_data, data_x, data_y):
	hour_flow = 24*20 - CARE_HOUR
	#add bias
	data_x[:, 0] = 1.0
	for month in range(12):
		for hour_start in range(hour_flow):
			for item in range(CARE_ITEM):
				for hour in range(CARE_HOUR):
					data_x[hour_flow*month+hour_start][item*CARE_HOUR + hour + 1] = train_data[item][480*month+hour_start+hour]
			data_y[hour_flow*month+hour_start] = train_data[9][480*month+hour_start+CARE_HOUR]

def valid(valid_x, valid_y, weight, fs_weight):
	my_y = my.cal_y(valid_x, weight, fs_weight)
	return my.cal_RMSE(my_y, valid_y)

def gen_batch(train_x, train_y, batch_x, batch_y):
	choice = np.random.choice(TRAIN_NUM, BATCH_SIZE, replace=False)
	for i in range(BATCH_SIZE):
		batch_x[i] = train_x[choice[i]]
		batch_y[i] = train_y[choice[i]]

def train(data_x, data_y, weight, fs_weight, i):
	last_time = False
	if i<0:
		last_time = True
	log_name = LOG_NAME + str(i) + '.txt'
	train_x = data_x[:TRAIN_NUM, :]
	train_y = data_y[:TRAIN_NUM]
	batch_x = np.zeros([BATCH_SIZE ,FEATURE_NUM], dtype=np.float64)
	batch_y = np.zeros([BATCH_SIZE], dtype=np.float64)	

	with open(log_name, 'a') as log:
		prev_gra = np.zeros((FEATURE_NUM,), dtype=np.float64)
		learn_rate = INIT_LR
		laps=0
		while(1):
			laps += 1
			if laps%10==0 and last_time==False:
				RMSE = valid(data_x[TRAIN_NUM:, :], data_y[TRAIN_NUM:], weight, fs_weight)
				log.write(str(laps) + '\t' + str(RMSE) + '\n')
			for i in range(int(TRAIN_NUM/BATCH_SIZE)):
				gen_batch(train_x, train_y, batch_x, batch_y)
				my_y = np.dot(batch_x, weight)
				dif = my_y-batch_y
				avg_lost = np.sum(np.square(dif))/BATCH_SIZE
				gra = np.dot(batch_x.transpose(), dif)
				prev_gra += gra**2
				ada = np.sqrt(prev_gra)
				weight -= learn_rate*gra/ada
			if abs(avg_lost) <= THR_LOST or laps >= THR_LAP:
				print('laps: ' + str(laps) + '\t' + 'avg_lost: ' + str(avg_lost))
				return

def opt_train(data_x, data_y, weight, fs_weight):
	for i in range(3):
		train(data_x, data_y, weight, fs_weight, i)
	TRAIN_NUM = DATA_NUM
	train(data_x, data_y, weight, fs_weight, -1)

def main():
	log_init(LOG_NAME)

	#read in train csv
	train_data = my.read_csv(TRAIN_NAME, 3, 27)

	#feature scaling for both data
	fs_weight = my.cal_weight(train_data)
	my.scale_down(train_data, fs_weight)
	np.savetxt(FWLOG_NAME, fs_weight, delimiter=',')

	#reshape and divide into x and y
	data_x = np.zeros([DATA_NUM ,FEATURE_NUM], dtype=np.float64)
	data_y = np.zeros([DATA_NUM], dtype=np.float64)
	reshape_train(train_data, data_x, data_y)

	#train & valid
	weight = np.full((FEATURE_NUM,), INIT_W, dtype=np.float64)
	opt_train(data_x, data_y, weight, fs_weight)

	np.savetxt(WLOG_NAME, weight, delimiter=',')

if __name__ == "__main__":
    main()
