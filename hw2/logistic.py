import csv
import sys
import numpy as np
import my_function as func
np.set_printoptions(threshold=np.nan)

FEATURE_NUM = func.FEATURE_NUM
FWLOG_NAME = func.FWLOG_NAME
WLOG_NAME = func.WLOG_NAME
LOG_NAME = func.LOG_NAME
TRAIN_X = sys.argv[1]
TRAIN_Y = sys.argv[2]

EXP_NUM = 0
DATA_NUM = 32561
VALID_NUM = 3000
TRAIN_NUM = DATA_NUM - VALID_NUM
BATCH_SIZE = 100
ELTA = 0.5
INIT_W = 0.5
THR_EPOCH = 1000

def log_init(log_name):
	with open(log_name, 'w') as log:        
		log.write('train/valid: ' + str(TRAIN_NUM) + '/'  + str(VALID_NUM) + '\n')
		log.write('features: ' + str(FEATURE_NUM) + '\n')
		log.write('learn_rate: ' + str(ELTA) + '\n')
		log.write('init_weight: ' + str(INIT_W) + '\n')
		log.write('epoch: ' + str(THR_EPOCH) + '\n')

def cal_acc(my_y, val_y):
	hit = 0
	for a, b in zip(my_y, val_y):
		if a == b:	
			hit += 1
	return float(hit)/float(val_y.shape[0])

def data_shuffle(data_x, data_y):
	order = np.arange(data_x.shape[0])
	np.random.shuffle(order)
	tmp_x = np.copy(data_x)
	tmp_y = np.copy(data_y)
	for i in range(data_x.shape[0]):
		data_x[i] = tmp_x[order[i]]
		data_y[i] = tmp_y[order[i]]

def gen_batch(data, i):
	if i != int(TRAIN_NUM/BATCH_SIZE):
		batch = data[(i*BATCH_SIZE):((i+1)*BATCH_SIZE)]
	else:
		batch = data[(i*BATCH_SIZE):]
	return batch

def train(data_x, data_y, weight, prev_gra):
	max_i = int(TRAIN_NUM/BATCH_SIZE)
	if TRAIN_NUM%BATCH_SIZE != 0:
		max_i += 1
	for i in range(max_i):
		batch_x = gen_batch(data_x, i)
		batch_y = gen_batch(data_y, i)
		z = np.dot(batch_x, weight)
		my_y = func.sigmoid(z)
		gra = 2*np.dot(batch_x.transpose(), (my_y-batch_y))
		prev_gra += gra**2
		sigma = np.sqrt(prev_gra)
		weight -= ELTA*gra/sigma

def validate(data_x, data_y, weight, epoch):
	if epoch%10 != 0 or TRAIN_NUM == DATA_NUM:
		return -1
	z = np.dot(data_x, weight)
	my_y = func.sigmoid(z)
	my_y_ = np.around(my_y)
	acc = cal_acc(my_y_, data_y)
	return acc

def training_process(data_x, data_y, fs_weight):
	for j in range(EXP_NUM + 1):
		global TRAIN_NUM
		#log_name = LOG_NAME + str(((j+1)%(EXP_NUM+1))) + '.txt'
		#print(log_name)
		if j == EXP_NUM:
			TRAIN_NUM = DATA_NUM

		#initialize
		weight = np.full((FEATURE_NUM,), INIT_W, dtype=np.float64)
		prev_gra = np.zeros((FEATURE_NUM, ), dtype=np.float64)
		epoch = 0

		while(1):
			epoch += 1
			data_shuffle(data_x, data_y)
			train(data_x[:TRAIN_NUM, :], data_y[:TRAIN_NUM], weight, prev_gra)
			if epoch >= THR_EPOCH:
				break
		'''
		with open(log_name, 'w') as log:
			while(1):
				epoch += 1
				data_shuffle(data_x, data_y)
				train(data_x[:TRAIN_NUM, :], data_y[:TRAIN_NUM], weight, prev_gra)
				acc = validate(data_x[TRAIN_NUM:, :], data_y[TRAIN_NUM:], weight ,epoch)
				if acc > 0:
					log.write(str(epoch) + "\t%.6f\n"%acc)
				if epoch >= THR_EPOCH:
					with open(LOG_NAME, 'a') as all_log:
						print("%.6f"%acc)
						all_log.write("%.6f\n"%acc)
					break
		'''
		if j == EXP_NUM:
			np.savetxt(WLOG_NAME, weight, delimiter=',')

def main():
	#log_init(LOG_NAME)
	
	#read csv
	train_x = (func.read_csv(TRAIN_X)).T
	train_y = (func.read_csv(TRAIN_Y))

	#feature_scaling
	fs_weight = func.cal_weight(train_x)
	func.scale_down(train_x, fs_weight)
	np.savetxt(FWLOG_NAME, fs_weight, delimiter=',')

	#training
	train_x = np.insert(train_x.T, 0, 1, axis=1)
	training_process(train_x, train_y, fs_weight)

if __name__ == "__main__":
	main()
