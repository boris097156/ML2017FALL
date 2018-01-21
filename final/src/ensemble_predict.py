#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np
import argparse
from gensim.models.word2vec import Word2Vec
import logging
from scipy.spatial.distance import cosine

#debug = __import__('IPython').core.debugger.Tracer()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def cal_frequency(argv):
    word_freq = {}
    word_amount = 0
    for i in range(5):
        file_name = os.path.join(argv.data_dir, '%d_train.txt' % (i+1))
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                for l in line:
                    word_amount += 1
                    if l in word_freq:
                        word_freq[l] += 1
                    else:
                        word_freq[l] = 1
    #count -> frequency
    for word in word_freq:
        word_freq[word] /= word_amount
    return word_freq

def dumpPrediction(filepath, predicts):
    with open(filepath, 'w') as f:
        f.write('id,ans\n')
        for idx, predict in enumerate(predicts):
            # NOTE: id starts from 1
            f.write('{},{}\n'.format(idx+1, predict))

def sentence2vector(sentence, word2vec, argv, word_freq):
    vec = np.zeros(argv.word_dim)
    count = 0
    a = 0.0035
    for word in sentence:
        if word not in word2vec.wv.vocab:
            continue
        vec += word2vec.wv[word]*(a/(a+(word_freq[word])))
        count += 1
    if count == 0:
        return vec
    return vec/count

def computeSimilarity(vec0, vec1):
    return 1 - cosine(vec0, vec1)

def predictIndex(dialogue, options):
    max_similarity, max_idx = None, None
    s = []
    for idx, option in enumerate(options):
        similarity = computeSimilarity(dialogue, option)
        s.append(similarity)
        if not max_similarity or similarity > max_similarity:
            max_similarity, max_idx = similarity, idx
    return s

def read_cut(file_name):
    dio = []
    opts = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        i = 0
        tmp = []
        for line in lines:
            line = line.strip().split()
            if i == 0:
                dio.append(line)
            else:
                tmp.append(line)
            if i == 6:
                opts.append(tmp)
                tmp = []
            i = (i+1)%7
    return dio, opts

def model_predict(word2vec, dialogues, options, word_freq):
    similarities = []

    all_sentences = []
    all_dialogue_vec = []
    all_options_vec = []
    
    for dialogue, options in list(zip(dialogues, options)):
        dialogue_vec = sentence2vector(dialogue, word2vec, argv, word_freq)
        options_vec = [sentence2vector(option, word2vec, argv, word_freq) for option in options]
        all_dialogue_vec.append(dialogue_vec)
        all_options_vec.append(options_vec)
        all_sentences.append(dialogue_vec)
        for i in range(6):
            all_sentences.append(options_vec[i])

    all_sentences = np.asarray(all_sentences).T
    U, s, V = np.linalg.svd((all_sentences), 0)

    for i in range(len(all_dialogue_vec)):
        dialogue_vec = np.array(all_dialogue_vec[i])
        options_vec = np.array(all_options_vec[i])

        dialogue_vec = dialogue_vec - np.dot(np.dot(U[0],np.transpose(U[0])),dialogue_vec)
        for j in range(6):
            options_vec[j] = options_vec[j] - np.dot(np.dot(U[0],np.transpose(U[0])),options_vec[j])
        similarity = predictIndex(dialogue_vec, options_vec)
        similarities.append(similarity)
    return similarities

def fair_voting(sim, model_num):
    predicts = []
    for i in range(len(sim[0])):
        voting = np.zeros(6)
        for j in range(model_num):
            max_sim = None
            max_indx = None
            for k in range(6):
                if not max_sim or sim[j][i][k] > max_sim:
                    max_sim = sim[j][i][k]
                    max_indx = k
            voting[max_indx] += 1
        predicts.append(np.argmax(voting))
    return predicts

def main(argv):

    model_num = argv.model_num
    word2vecs = []
    word2vec1 = Word2Vec.load(argv.load_word2vec1)
    word2vecs.append(word2vec1)
    if model_num >= 2:
        word2vec2 = Word2Vec.load(argv.load_word2vec2)
        word2vecs.append(word2vec2)
        if model_num >= 3:
            word2vec3 = Word2Vec.load(argv.load_word2vec3)
            word2vecs.append(word2vec3)
            if model_num >= 4:
                word2vec4 = Word2Vec.load(argv.load_word2vec4)
                word2vecs.append(word2vec4)
                if model_num >= 5:
                    word2vec5 = Word2Vec.load(argv.load_word2vec5)
                    word2vecs.append(word2vec5)
                    if model_num >= 6:
                        word2vec6 = Word2Vec.load(argv.load_word2vec6)
                        word2vecs.append(word2vec6)
                        if model_num >= 7:
                            word2vec7 = Word2Vec.load(argv.load_word2vec7)
                            word2vecs.append(word2vec7)    

    #seg_test_path = os.path.join(argv.data_dir, argv.seg_test)
    dialogues, options = read_cut(argv.seg_test)
    #word_freq = cal_frequency(argv)
    word_freq = np.load('../model/word_freq.npy').item()

    sim = []
    for i in range(model_num):
        sim1 = model_predict(word2vecs[i], dialogues, options, word_freq)
        sim.append(sim1)

    #predicts = max_similarity(sim, model_num)
    predicts = fair_voting(sim, model_num)
    #predicts = avg_similarity(sim, model_num)

    dumpPrediction(argv.output_csv, predicts)

def parseArgv():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('--load_word2vec1', type=str, required=True)
    parser.add_argument('--load_word2vec2', type=str, required=False)
    parser.add_argument('--load_word2vec3', type=str, required=False)
    parser.add_argument('--load_word2vec4', type=str, required=False)
    parser.add_argument('--load_word2vec5', type=str, required=False)
    parser.add_argument('--load_word2vec6', type=str, required=False)
    parser.add_argument('--load_word2vec7', type=str, required=False)
    parser.add_argument('--word_dim', type=int, required=True)
    parser.add_argument('--model_num', type=int, required=True)
    parser.add_argument('--seg_test', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    parser.add_argument('--data_dir', type=str, default='./')
    return parser.parse_args()

if __name__ == '__main__':
    argv = parseArgv()
    main(argv)
