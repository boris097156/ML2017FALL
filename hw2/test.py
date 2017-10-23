import csv
import sys
import numpy as np
import my_function as func

TEST_X = sys.argv[1]
OUTPUT_NAME = sys.argv[2]
FWLOG_NAME = func.FWLOG_NAME
WLOG_NAME = func.WLOG_NAME

def main():
	test_x = (func.read_csv(TEST_X)).T
	
	fs_weight = np.loadtxt(FWLOG_NAME, delimiter=',')
	func.scale_down(test_x, fs_weight)
	test_x = np.insert(test_x.T, 0, 1, axis=1)
	
	weight = np.loadtxt(WLOG_NAME, delimiter=',')
	z = np.dot(test_x, weight)
	my_y = func.sigmoid(z)
	my_y_ = np.around(my_y)
	
	func.csv_result(my_y_, OUTPUT_NAME)

if __name__ == '__main__':
	main()
