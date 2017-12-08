import csv
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
from keras.models import Sequential, load_model
import numpy as np
import my_function as func

SG = 1
TYPE = func.TYPE
#DATA_DIR = 'data/'
#MODEL_DIR = './'
#TEST_DATA = DATA_DIR + 'testing_data.txt'
TEST_DATA = sys.argv[1]
SUBMISSION_FILE = sys.argv[2]
MODEL_NAME = TYPE + '.h5'
#MODEL_NAME = MODEL_DIR + 'model' + str(SG) + str(MODEL_V) +'.h5'
#CHECK_NAME = MODEL_DIR + 'check_model' + str(SG) + str(MODEL_V) + '.h5'

def test_model():
        test_x, label = func.read_data(TEST_DATA)
        print(TYPE)
        if TYPE == 'without':
            print('without checked')
            test_x = func.remove_punctuation(test_x)
        test_x = func.replace_digits(test_x)
        test_x = func.word2vec(test_x, int(SG))

        #model = load_model(CHECK_NAME)
        model = load_model(MODEL_NAME)
        ans = model.predict(test_x)
        ans = np.around(ans).reshape(ans.shape[0],)

        with open(SUBMISSION_FILE, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
                writer.writerow(['id']+['label'])
                for i in range(ans.shape[0]):
                        writer.writerow([str(i)]+[str(int(ans[i]))])

def main():
        test_model()

if __name__ == '__main__':
        main()
