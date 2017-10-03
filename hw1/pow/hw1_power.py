import csv
import math
import numpy as np

'''
******try******
validation sets
stochastic
learn_rate
hazard
more features
more power(次方)
'''

feature_weight = [[] for _ in range(18)]

def feature_scaling(data_size, my_data, flag):
	for i in range(18):
		if flag == 0:
			my_sum = 0.0;
			for j in range(data_size):
				my_sum += my_data[i][j]
			my_avg = my_sum/data_size
			my_squ = 0.0;
			for j in range(data_size):
				my_squ += (my_data[i][j]-my_avg)**2
			my_sd = (my_squ/data_size)**0.5
			for j in range(data_size):
				my_data[i][j] -= my_avg
				my_data[i][j] /= my_sd
			feature_weight[i].append(my_avg)
			feature_weight[i].append(my_sd)
		elif flag == 1:
			for j in range(data_size):
				my_data[i][j] -= feature_weight[i][0];
				my_data[i][j] /= feature_weight[i][1];
	return

train_Data = [[] for _ in range(18)]

# Read in train.csv to list of 18 lists, every sublist represents an item.
with open('train.csv', 'r', encoding='ISO-8859-1') as csv_file:
	raw_file = csv.reader(csv_file , delimiter=",")
	n_row = 0;
	for row in raw_file:
		if n_row != 0:
			for item in range(3,27):
				if row[item] != "NR":
					train_Data[(n_row-1)%18].append( float( row[item] ) )
				else:
					train_Data[(n_row-1)%18].append( float( 0 ) )	
		n_row += 1

#feature scaling of train_Data(18,5760)
feature_scaling(5760, train_Data, 0)

np.savetxt('tmp2', train_Data, delimiter=',')

train_x = []
train_y = []

#setup basic variables
FEATURE_NUM = 325 	#1+18*9*2
weight = np.full((FEATURE_NUM,), 0.5, dtype=np.float32)	#shape of (FEATURE_NUM,)

#Reshape train_Data(18,5760) to train_x(5652,FEATURE_NUM) and train_y.(5652,)
for month in range(12):
	for hour_start in range(471):
		train_x.append([1])
		for item in range(18):
			for hour in range(9):
				train_x[471*month+hour_start].append(train_Data[item][480*month+hour_start+hour])
				train_x[471*month+hour_start].append(pow(train_Data[item][480*month+hour_start+hour],2))
		print(train_Data[9][480*month+hour_start+9])
		train_y.append(train_Data[9][480*month+hour_start+9])

#5652 = amount of training datas
train_x = np.array(train_x)  #shape of (5652,FEATURE_NUM)
train_y = np.array(train_y)  #shape of (5652,)
#train
prev_gra = np.zeros((FEATURE_NUM,), dtype=np.float32)
laps = 0
learn_rate = 0.5
data_amount = 5652

with open('log_pow.txt', 'w') as log:
	while(1):
		my_y = np.dot(train_x, weight)
		dif = my_y-train_y
		#lost = np.sum(dif)
		avg_lost = np.sum(np.square(dif))/data_amount
		#print(avg_lost)
		log.write(str(laps) + '\t' + str(avg_lost) + '\n')
	#	if laps%30000 == 0:
	#		learn_rate = learn_rate/10
		if abs(avg_lost) <= 0.1 or laps>12000:
			break;
		gra = np.dot(train_x.transpose(), dif)/data_amount
		prev_gra += gra**2
		ada = np.sqrt(prev_gra)
		weight -= learn_rate*gra/ada
		laps += 1 
	log.write('laps: ' + str(laps) + '\tavg_lost: ' + str(avg_lost))

#Read in test_X.csv to list of 18 lists, every sublist represents an item.
test_Data = [[] for _ in range(18)]
with open('test.csv', 'r', encoding='ISO-8859-1') as csv_file:
	raw_file = csv.reader(csv_file , delimiter=",")
	n_row = 0;
	for row in raw_file:
		for item in range(2,11):
			if row[item] != "NR":
				test_Data[(n_row)%18].append( float( row[item] ) )
			else:
				test_Data[(n_row)%18].append( float( 0 ) )	
		n_row += 1

#feature scaling of test_Data(18,2160)
feature_scaling(2160, test_Data, 1)

#Reform from (18,2160) to (240, 1+162)
test_x = []
for test_id in range(240):
	test_x.append([1])
	for item in range(18):
		for hour in range(9):
			test_x[test_id].append(test_Data[item][test_id*9+hour])
			test_x[test_id].append(pow(test_Data[item][test_id*9+hour],2))
test_x = np.array(test_x) #shape of (240,FEATURE_NUM)

#Testing
test_y = np.dot(test_x, weight) #shape of (240,)

#Return value of PM2.5
for i in range(240):
	test_y[i] *= feature_weight[9][1]
	test_y[i] += feature_weight[9][0]

with open('submission.csv', 'w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['id']+['value'])
	for i in range(240):
		writer.writerow(['id_'+str(i)]+[str(test_y[i])])
