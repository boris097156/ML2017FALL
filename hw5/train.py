import csv
import os
import sys
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Flatten
from keras.layers import Dot, Add, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K

EMBEDDING_DIM = 361
USER_AMOUNT = 6041
MOVIE_AMOUNT = 3883
EPOCH = 500
MODEL_NAME = 'best.h5'
CHECK_NAME = 'check.h5'

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true)**2))

def read_data(file_name):
    users=[]
    movies=[]
    rating=[]
    is_train = False
    if file_name.find('train') >= 0:
        is_train = True
    with open(file_name, 'r', encoding='ISO-8859-1') as f:
        f.readline()
        lines = f.readlines()
        for row in lines:
            row = row.strip().split(',')
            users.append(row[1])
            movies.append(row[2])
            if is_train == True:
                rating.append(float(row[3]))
    if is_train == True:
        return np.asarray(users), np.asarray(movies), np.asarray(rating)
    else:
        return np.asarray(users), np.asarray(movies)

def normalize(ratings):
    ratings = (ratings-1)/4
    return ratings

def my_shuffle(users, movies, ratings):
    order = np.arange(users.shape[0])
    np.random.shuffle(order)
    u = []
    m = []
    r = []
    for i in order:
        u.append(users[i])
        m.append(movies[i])
        r.append(ratings[i])
    return np.asarray(u), np.asarray(m), np.asarray(r)

def train():
    users, movies, ratings = read_data('train.csv')
    max_user = int(np.amax(np.array(users, dtype='int')))
    max_movie = int(np.amax(np.array(movies, dtype='int')))
    users, movies, ratings = my_shuffle(users, movies, ratings)
    user = Input(shape=(1,))
    u = Embedding(max_user+1, EMBEDDING_DIM, embeddings_initializer='glorot_normal')(user)
    u = Flatten()(u)
    u = Dropout(0.5)(u)
    movie = Input(shape=(1,))
    m = Embedding(max_movie+1, EMBEDDING_DIM, embeddings_initializer='glorot_normal')(movie)
    m = Flatten()(m)
    m = Dropout(0.5)(m)
    dot = Dot(axes=-1)([u, m])
    u_bias = Embedding(max_user+1, 1, embeddings_initializer='zero')(user)
    u_bias = Flatten()(u_bias)
    m_bias = Embedding(max_movie+1, 1, embeddings_initializer='zero')(movie)
    m_bias = Flatten()(m_bias)
    out = Add()([dot, u_bias, m_bias])
    model = Model([user, movie], out)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00005), metrics=[RMSE])
    #plot_model(model, to_file='model.png', show_shapes=True)
    earlyStopping = EarlyStopping(monitor='val_RMSE', patience=5, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(CHECK_NAME, monitor='val_RMSE', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    history = model.fit([users, movies], ratings, callbacks=[earlyStopping, checkpoint] ,batch_size=32, shuffle=True, verbose=2, epochs=EPOCH, validation_split=0.1)
    model.save(MODEL_NAME)
    '''
    plt.plot(history.history['RMSE'])
    plt.plot(history.history['val_RMSE'])
    plt.title('model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('RMSE.png')
    '''

def test():
    users, movies = read_data('test.csv')
    model = load_model(MODEL_NAME, custom_objects={'RMSE': RMSE})
    ans = model.predict([users, movies])
    ans = ans.reshape(ans.shape[0],)
    ans = np.clip(ans, 1, 5)

    with open(SUBMISSION_FILE, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        writer.writerow(['TestDataID']+['Rating'])
        for i in range(ans.shape[0]):
            writer.writerow([str(i+1)]+[str(ans[i])])

def main():
    train()
    #test()

if __name__ == '__main__':
    main()
