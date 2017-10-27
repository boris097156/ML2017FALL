import numpy as np
import matplotlib.pyplot as plt
import my_function as func
import sys

from sklearn import ensemble

TRAIN_X = sys.argv[1]
TRAIN_Y = sys.argv[2]
TEST_X = sys.argv[3]
OUTPUT = sys.argv[4]

train_x = (func.read_csv(TRAIN_X)).T
train_y = (func.read_csv(TRAIN_Y))
test_x = (func.read_csv(TEST_X)).T
all_x = np.concatenate([train_x, test_x], axis=1)

fs_weight = func.cal_weight(all_x)
func.scale_down(train_x, fs_weight)
func.scale_down(test_x, fs_weight)

train_x = train_x.T
test_x = test_x.T

clf = ensemble.GradientBoostingClassifier(loss='exponential', n_estimators=800, learning_rate=0.1, max_depth=4, random_state=0).fit(train_x, train_y)

test_y = clf.predict(test_x)
func.csv_result(test_y, OUTPUT)
