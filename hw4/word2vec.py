import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import gensim, logging
import my_function as func
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SG = 1
DATA_DIR = './'
MODEL_DIR = './'
TYPE = func.TYPE
#TRAIN_LABEL = DATA_DIR + 'training_label.txt'
#TRAIN_NOLABEL = DATA_DIR + 'training_nolabel.txt'
#TEST_DATA = DATA_DIR + 'testing_data.txt'
TRAIN_LABEL = sys.argv[1]
TRAIN_NOLABEL = sys.argv[2]
MERGED_DATA = DATA_DIR + 'merged.txt'
MODEL_PATH = MODEL_DIR + 'word2vec_' + TYPE + '.model'

def create_merge():
    f = open(MERGED_DATA, 'w')
    f.close()

def word2vec():
    sentences = gensim.models.word2vec.Text8Corpus(MERGED_DATA)
    model = gensim.models.Word2Vec(sentences, workers=4, size=200, min_count=0, sg=SG, iter=10)
    model.save(MODEL_PATH)

def merge(data):
    sentences, y = func.read_data(data)
    print(TYPE)
    if TYPE == 'without':
        print('without checked')
        sentences = func.remove_punctuation(sentences)
    sentences = func.replace_digits(sentences)
    func.merge_data(sentences, MERGED_DATA)

def main():
    create_merge()
    merge(TRAIN_LABEL)
    merge(TRAIN_NOLABEL)
    #merge(TEST_DATA)
    word2vec()

if __name__ == '__main__':
    main()
