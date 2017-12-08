import csv
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gensim, logging
import my_function as func

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SG = 1
EPOCHS = 15
MAX_LENGTH = func.MAX_LENGTH
MODEL_DIR = './'
TRAIN_LABEL = sys.argv[1]

TYPE = func.TYPE
MODEL_NAME = TYPE + '.h5'
CHECK_NAME =  MODEL_DIR + 'check_' + TYPE + str(SG) + '.h5'

earlyStopping = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(CHECK_NAME, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def train_RNN():
    sentences, y = func.read_data(TRAIN_LABEL)
    print(TYPE)
    if TYPE == 'without':
        print('without checked')
        sentences = func.remove_punctuation(sentences)
    sentences = func.replace_digits(sentences)
    sentences = func.word2vec(sentences ,SG)
    model = Sequential()
    model.add(LSTM(128, input_shape=(MAX_LENGTH, 200), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(128, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(sentences, y, callbacks=[earlyStopping, checkpoint], batch_size=32, epochs=EPOCHS, validation_split=0.1, verbose=2)
    model.save(MODEL_NAME)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(TYPE + '.png')

def main():
    train_RNN()

if __name__ == '__main__':
    main()
