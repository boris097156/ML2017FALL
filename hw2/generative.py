import csv
import sys
import numpy as np
import my_function as func
np.set_printoptions(threshold=np.nan)

TRAIN_X = sys.argv[1]
TRAIN_Y = sys.argv[2]
TEST_X = sys.argv[3]
OUTPUT_NAME = sys.argv[4]

def cal_mean(train_x, train_y, flag):
	mu = np.zeros((train_x.shape[1],))
	count = 0
	for i in range(train_x.shape[0]):
		if train_y[i] == flag:
			mu += train_x[i]
			count += 1
	mu /= count
	return mu, count

def cal_sigma(train_x, train_y, mu, count, flag):
	sigma = np.zeros((train_x.shape[1], train_x.shape[1]))
	for i in range(train_x.shape[0]):
		if train_y[i] == flag:
			sigma += np.dot(np.transpose([train_x[i] - mu]), [(train_x[i] - mu)])
	sigma /= count
	return sigma

def cal_statics(train_x, train_y):
	train_size = train_x.shape[0]
	mu0, count0 = cal_mean(train_x, train_y, 1)
	mu1, count1 = cal_mean(train_x, train_y, 0)
	sigma0 = cal_sigma(train_x, train_y, mu0, count0, 1)
	sigma1 = cal_sigma(train_x, train_y, mu1, count1, 0)
	shared_sigma = (float(count0)/train_size)*sigma0 + (float(count1)/train_size)*sigma1
	return mu0, mu1, shared_sigma, float(count0)/float(count1)

def predict(test_x, mu0, mu1, sigma, ratio):
	sigma_inverse = np.linalg.inv(sigma)
	w = np.dot((mu0-mu1), sigma_inverse)
	b = np.dot(np.dot([mu0], sigma_inverse), mu0)/(-2) + np.dot(np.dot([mu1], sigma_inverse), mu1)/2 + np.log(ratio)
	f = np.dot(w, test_x) + b
	y = func.sigmoid(f)
	my_y = np.around(y)
	return my_y

def main():
	#read train csv
	train_x = (func.read_csv(TRAIN_X)).T
	train_y = (func.read_csv(TRAIN_Y))
	test_x = (func.read_csv(TEST_X)).T

	#feature_scaling
	all_x = np.concatenate([train_x, test_x], axis=1)
	fs_weight = func.cal_weight(all_x)
	func.scale_down(train_x, fs_weight)
	func.scale_down(test_x, fs_weight)

	#calculate statics	
	mu0, mu1, shared_sigma, ratio = cal_statics(train_x.T, train_y)
	my_y = predict(test_x, mu0, mu1, shared_sigma, ratio)	
	func.csv_result(my_y, OUTPUT_NAME)

if __name__ == "__main__":
	main()
