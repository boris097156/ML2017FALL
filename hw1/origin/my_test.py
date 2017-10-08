import csv
import sys
import numpy as np
import my_function as my

INPUT_NAME = sys.argv[1]
OUTPUT_NAME = sys.argv[2]
CARE_ITEM = my.CARE_ITEM
CARE_HOUR = my.CARE_HOUR

#Paths
FEATURE_NUM = my.FEATURE_NUM
WLOG_NAME = my.WLOG_NAME
FWLOG_NAME = my.FWLOG_NAME
hour_start = 9 - CARE_HOUR

def reshape_test(test_data, test_x):
	test_x[:, 0] = 1
	for i in range(240):
		for item in range(CARE_ITEM):
			for hour in range(CARE_HOUR):
				test_x[i][item*CARE_HOUR+hour+1] = test_data[item][i*9+hour_start+hour]

def csv_result(test_y):
	with open(OUTPUT_NAME, 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(['id']+['value'])
		for i in range(240):
			writer.writerow(['id_'+str(i)]+[str(test_y[i])])

def main():
	#read in test.csv
	test_data = my.read_csv(INPUT_NAME, 2, 11)
	
	#feature scaling
	fs_weight = np.loadtxt(FWLOG_NAME, delimiter=',').reshape(CARE_ITEM,2)
	my.scale_down(test_data, fs_weight)
	
	#reshape test
	test_x = np.zeros([240, FEATURE_NUM], dtype=np.float64)
	reshape_test(test_data, test_x)

	weight = np.loadtxt(WLOG_NAME, delimiter=',')
	
	#test & scale up
	test_y = my.cal_y(test_x, weight, fs_weight)

	csv_result(test_y)

if __name__ == "__main__":
	main()
