import csv
import sys
import numpy as np
import my_fs as fs
import my_train as train

INPUT_NAME = sys.argv[1]
OUTPUT_NAME = sys.argv[2]
ITEM_NUM = train.ITEM_NUM
FEATURE_NUM = train.FEATURE_NUM
WLOG_NAME = train.WLOG_NAME
FWLOG_NAME = train.FWLOG_NAME

def reshape_test(test_data, test_x):
	for i in range(240):
		test_x.append([1])
		for item in range(ITEM_NUM):
			for hour in range(9):
				test_x[i].append(test_data[item][i*9+hour])

def log_result(test_y):
	with open(OUTPUT_NAME, 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(['id']+['value'])
		for i in range(240):
			writer.writerow(['id_'+str(i)]+[str(test_y[i])])

def main():
	#read in test.csv
	test_data = [[] for _ in range(ITEM_NUM)]
	train.read_csv(INPUT_NAME, test_data, 2, 11)

	#feature scaling
	fs_weight = np.loadtxt(FWLOG_NAME, delimiter=',')
	fs.scale_down(test_data, fs_weight)
	
	#reshape test
	test_x = []
	reshape_test(test_data, test_x)
	test_x = np.array(test_x)
	
	#read in weight
	weight = np.loadtxt(WLOG_NAME, delimiter=',')
	
	#test & scale up
	test_y = np.dot(test_x, weight)
	fs.scale_up(test_y, fs_weight)

	log_result(test_y)

if __name__ == "__main__":
	main()
'''
#Reform from (18,2160) to (240, 1+162)
test_x = []
for test_id in range(240):
    test_x.append([1])
    for item in range(18):
        for hour in range(9):
            test_x[test_id].append(test_Data[item][test_id*9+hour])
test_x = np.array(test_x)
#Testing
test_y = np.dot(test_x, weight) #shape of (240,)

#Return value of PM2.5
for i in range(240):
    test_y[i] *= fs_weight[9][1]
    test_y[i] += fs_weight[9][0]

with open('submission.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id']+['value'])
    for i in range(240):
        writer.writerow(['id_'+str(i)]+[str(test_y[i])])
'''
