import os
import re
import argparse
import numpy as np
import jieba

def jieba_setting(dict_path, stopwords_path=None):
    # set up the dictionary for word segmentation and read the stopwords
    # for Simplified Chinese: dict='dict.txt', stopwords='SCstopwords.txt'
    # usage:
    # jieba_dir = 'jieba_dict'
    # dict_path = os.path.join(jieba_dir, 'dict.txt')
    # stopwords_path = os.path.join(jieba_dir, 'stopwords.txt')
    # stopwords = jieba_setting(dict_path, stopwords_path)
    jieba.set_dictionary(dict_path)
    stopwords = []
    if stopwords_path != None:
        with open(stopwords_path,'r') as f:
            for line in f:
                stopwords.append(line.strip())
    return stopwords

def clean_line(line):
    line = line.strip()
    line = line.replace(" ", "")
    line = re.sub('\W', '', line) # remove symbols
    line = re.sub(r'[a-zA-Z0-9]+', '', line)
    line = re.sub(r'[\uff21-\uff3a]+', '', line) # remove full width character
    return line

def cut_train(dict_path, train_dir, output_dir, stopwords=None, sc=True):
    # cut files in the train_dir and write results in the output_dir
    # usage:
    # dict_path = 'jieba_dict/dict.txt'
    # train_dir = 'data/training_data'
    # output_dir = 'data/SC_seg_data'
    # cut_train(dict_path, train_dir, output_dir, stopwords=None, sc=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if sc:
        from opencc.opencc import OpenCC
        openCC = OpenCC('tw2s')
    jieba_setting(dict_path)
    for file in os.listdir(train_dir):
        output = open(os.path.join(output_dir, file), 'w')
        with open(os.path.join(train_dir, file), 'r') as f:
            for line in f:
                line = clean_line(line)
                if sc:
                    line = openCC.convert(line)
                words = jieba.cut(line, cut_all=False)
                if stopwords:
                    line= [word for word in words if word not in stopwords]
                else:
                    line= [word for word in words]
                if len(line) > 0:
                    print(' '.join(line), file=output)
        output.close()

def parse_and_cut_test(dict_path, test_path, output_dir, sc=True):
    # Parse 'testing_data.csv', cut it and store the result into a file
    # usage:
    # parse_and_cut_test('jieba_dict/dict.txt', 'data/testing_data.csv', 'data/SC_seg_data', sc=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if sc:
        from opencc.opencc import OpenCC
        openCC = OpenCC('tw2s')
    jieba_setting(dict_path)
    with open(test_path , 'r' , encoding = 'utf-8') as f:
        output_path = os.path.join(output_dir, 'SC_seg_test.txt')
        print(output_path)
        output_f = open(output_path, 'w')
        
        lines = f.read().strip().split('\n')[1:]
        for line in lines:
            test_id , diologue , options = [i for i in line.split(',')]
            diologue = clean_line(diologue)
            if sc:
                diologue = openCC.convert(diologue)
            diologue = jieba.cut(diologue, cut_all=False)
            print(' '.join(diologue), file=output_f)
            options = options.strip().split('\t')
            for i in range(6):
                option = clean_line(options[i])
                if sc:
                    option = openCC.convert(option)
                option = jieba.cut(option, cut_all=False)
                print(' '.join(option), file=output_f)

def main(args):
    dict_path = '../jieba_dict/dict.txt'
#     cut_train(dict_path, args.train_dir, args.output_dir, stopwords=None, sc=True)
    parse_and_cut_test(dict_path, args.test_path, args.output_dir, sc=True)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='data/SC_seg_data',
                        help='Path to processed data directory')
    
    main(parser.parse_args())