import numpy as np
import string
import re
import gensim, logging
from keras.preprocessing.sequence import pad_sequences

MODEL_DIR = './'
MAX_LENGTH = 36
TYPE = 'with'
preserved_set = string.ascii_letters + string.digits + ' '

def remove_punctuation(sentences):
	new_sentences = []
	for row in sentences:
		row = "".join(c for c in row if c in preserved_set)
		row = row.strip().split()
		row = " ".join(word for word in row)
		new_sentences.append(row)
	return new_sentences

def replace_digits(sentences):
	new_sentences = []
	for row in sentences:
		row = re.sub(r'\d+', '1', row)
		new_sentences.append(row)
	return new_sentences

def read_data(file_name):
	x = []
	y = []
	with open(file_name, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		if file_name.find('test') >= 0:
			lines = lines[1:]
		for row in lines:
			row = row.strip('\n')
			if file_name.find('nolabel') < 0:
				row = row.replace(',', ' , ', 1)
				row = row.strip().split()
				y.append(int(row[0]))
				row = " ".join(word for word in row[2:])
			x.append(row)
	return x, np.array(y)

def word2vec(sentences, sg):
    print(TYPE)
    w2v_model = 'word2vec_' + TYPE + '.model'
    print('loading w2v model')
    model = gensim.models.Word2Vec.load(w2v_model)
    print('finish loading w2v model')
    new_sentences = []
    print('transforming w2v')
    for sentence in sentences:
        sentence = sentence.strip().split()
        sentence = [model.wv[word] for word in sentence if word in model.wv.vocab]
        new_sentences.append(sentence)
    padded_docs = pad_sequences(new_sentences, dtype='float32', maxlen=MAX_LENGTH, padding='post')
    print('finish transforming w2v')
    return np.asarray(padded_docs)

def merge_data(labeled, merged):
	with open(merged, 'a', encoding='utf-8') as f:
		for row in labeled:
			f.write(row + '\n')
