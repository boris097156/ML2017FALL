import csv
import sys
import numpy as np
import my_function as my

#Paths
LOG_NAME = my.LOG_NAME
WLOG_NAME = my.WLOG_NAME
FWLOG_NAME = my.FWLOG_NAME
TRAIN_NAME = '../train.csv'

#feature variables
CARE_HOUR = my.CARE_HOUR
CARE_ITEM = my.CARE_ITEM
FEATURE_NUM = my.FEATURE_NUM
PM_POSITION = my.PM_POSITION

#training variables
ORIGIN_DATA_NUM = 24*20*12
DATA_NUM = ORIGIN_DATA_NUM - CARE_HOUR*12
VALID_NUM = int(sys.argv[1])
TRAIN_NUM = DATA_NUM - VALID_NUM
BATCH_SIZE = 100
ELTA = np.random.random_sample()/2
INIT_W = np.random.random_sample()/2
LANDA = 0.0001
THR_LAP = 7500
THR_RMSE = 0
LOG_NUM = my.LOG_NUM

def log_init(log_name):
	with open(log_name, 'w') as log:
		log.write('train/valid: ' + str(TRAIN_NUM) + '/'  + str(VALID_NUM) + '\n')
		log.write('features: ' + str(FEATURE_NUM) + '\n')
		log.write('care hour: ' + str(CARE_HOUR) + '\n')
		log.write('care item: ' + str(CARE_ITEM) + '\n')
		log.write('learn_rate: ' + str(ELTA) + '\n')
		log.write('init_weight: ' + str(INIT_W) + '\n')
		log.write('landa: ' + str(LANDA) + '\n')

def reshape_train(train_data, data_x, data_y):
	hour_flow = 24*20 - CARE_HOUR
	#add bias
	data_x[:, 0] = 1.0
	for month in range(12):
		for hour_start in range(hour_flow):
			for item in range(CARE_ITEM):
				for hour in range(CARE_HOUR):
					data_x[hour_flow*month+hour_start][item*CARE_HOUR + hour + 1] = train_data[item][480*month+hour_start+hour]
			data_y[hour_flow*month+hour_start] = train_data[PM_POSITION][480*month+hour_start+CARE_HOUR]

def cal_RMSE(my_y, small_y, fs_weight):
	real_y = np.copy(small_y)
	my.scale_up(real_y, fs_weight)
	RMSE = np.sqrt((np.sum(np.square(my_y-real_y))/real_y.shape[0]))
	return RMSE

def gen_batch(train_x, train_y, batch_x, batch_y):
	choice = np.random.choice(TRAIN_NUM, BATCH_SIZE, replace=False)
	for i in range(BATCH_SIZE):
		batch_x[i] = train_x[choice[i]]
		batch_y[i] = train_y[choice[i]]

def train(data_x, data_y, weight, fs_weight, i):
	no_valid = False
	if i<0:
		no_valid = True
		i = '_no_valid'
	log_name = LOG_NAME + str(i) + '.txt'
	train_x = data_x[:TRAIN_NUM, :]
	train_y = data_y[:TRAIN_NUM]
	batch_x = np.zeros([BATCH_SIZE ,FEATURE_NUM], dtype=np.float64)
	batch_y = np.zeros([BATCH_SIZE], dtype=np.float64)	

	with open(log_name, 'a') as log:
		prev_gra = np.zeros((FEATURE_NUM,), dtype=np.float64)
		eta = ELTA
		rmse = 1000
		laps = 0
		while(1):
			laps += 1
			for i in range(int(TRAIN_NUM/BATCH_SIZE)):
				gen_batch(train_x, train_y, batch_x, batch_y)
				my_y = np.dot(batch_x, weight)
				gra = 2*(np.dot(batch_x.transpose(), (my_y-batch_y)) + LANDA*weight)
				prev_gra += gra**2
				sigma = np.sqrt(prev_gra)
				weight -= eta*gra/sigma		
			#record validation's RMSE
			if laps%10==0:
				if no_valid == False:
					my_y = my.cal_y(data_x[TRAIN_NUM:, :], weight, fs_weight)	
					rmse = cal_RMSE(my_y, data_y[TRAIN_NUM:], fs_weight)
					log.write(str(laps) + '\t' + str(rmse) + '\n')
				else:
					my_y = my.cal_y(data_x[TRAIN_NUM:, :], weight, fs_weight)	
					rmse = cal_RMSE(my_y, data_y[TRAIN_NUM:], fs_weight)
					log.write(str(laps) + '\t' + str(rmse) + '\n')
			#record last RMSE
			if abs(rmse) <= THR_RMSE or laps >= THR_LAP:
				with open(LOG_NAME, 'a') as log_log:
					log_log.write('laps: ' + str(laps) + '\t' + 'RMSE: ' + str(rmse) + '\n')
				return

def opt_train(data_x, data_y, weight, fs_weight):
	for i in range(LOG_NUM):
		train(data_x, data_y, weight, fs_weight, i)
		weight = np.full((FEATURE_NUM,), INIT_W, dtype=np.float64)
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
