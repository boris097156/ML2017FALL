import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.models import load_model

IMG_DIR = sys.argv[1]
TEST_DIR = sys.argv[2]
SUB_DIR = sys.argv[3]

def do_pca(imgs):
	print('doing pca')
	pca = PCA(n_components=PCA_DIM)
	pca.fit(imgs)
	x = pca.transform(imgs)
	return x

def normalize(x):
	print('doing normalize')
	x = x/255
	x -= np.mean(x, axis=0)
	#print(x.shape)
	#print(np.mean(x, axis=0).shape)
	#sys.exit()
	return x

def do_tsne(x):
	print('doing tsne')
	x_embedded = TSNE(n_components=2).fit_transform(x)
	return x_embedded

def do_kmean(x):
	kmeans_fit = KMeans(n_clusters = 2).fit(x)
	cluster_labels = kmeans_fit.labels_
	return cluster_labels

def read_test():
	test_data = []
	with open(TEST_DIR, 'r') as csv_file:
		raw_file = csv.reader(csv_file, delimiter=",")
		next(raw_file)
		for row in raw_file:
			row = row[1:]
			test_data.append(row)
	return np.asarray(test_data, dtype=np.int)

def write_answer(answer):
	with open(SUB_DIR, 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
		writer.writerow(['ID']+['Ans'])
		for i in range(len(answer)):
			writer.writerow([str(i)]+[str(answer[i])])

def test(labels):
	test_data = read_test()
	answer =[]
	for i in range(test_data.shape[0]):
		if (labels[test_data[i][0]] == labels[test_data[i][1]]):
			answer.append(1)
		else:
			answer.append(0)
	write_answer(answer)

def build_encoder(input_img, encoding_dim):
	encoded = Dense(128, activation='relu')(input_img)
	encoded = Dense(64, activation='relu')(encoded)
	encoded = Dense(32, activation='relu')(encoded)
	return encoded

def build_decoder(encoder_output, input_dim):
	decoded = Dense(64, activation='relu')(encoder_output)
	decoded = Dense(128, activation='relu')(decoded)
	decoded_output = Dense(input_dim, activation='relu')(decoded)
	return decoded_output

def train(x):
	'''
	input_dim = 28*28
	input_img = Input(shape=(input_dim,))
	encoder_output = build_encoder(input_img, 2)
	encoder = Model(inputs=input_img, outputs=encoder_output)

	decoded = build_decoder(encoder_output, input_dim)
	autoencoder = Model(inputs=input_img, outputs=decoded)
	autoencoder.compile(optimizer='adam', loss='mse')
	#plot_model(autoencoder, to_file='model.png', show_shapes=True)
	earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
	autoencoder.fit(x, x, callbacks=[earlyStopping],  epochs=200, batch_size=128, shuffle=True, validation_split=0.1)
	'''
	encoder = load_model('encoder.h5')
	encoded_imgs = encoder.predict(x)
	#print(encoded_imgs.shape)
	return encoded_imgs

def main():
	x = np.load(IMG_DIR)
	x = normalize(x)
	#x = do_pca(x)
	#x = do_tsne(x)
	x = train(x)
	labels = do_kmean(x)
	test(labels)

if __name__ == '__main__':
	main()

