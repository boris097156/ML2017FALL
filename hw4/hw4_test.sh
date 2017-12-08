#! /bin/bash
echo "downloading models"
wget https://www.dropbox.com/s/qiqwn7835cs72mg/with.h5?dl=1 -O with.h5
wget https://www.dropbox.com/s/kj4i4pqc0dhiho3/word2vec_with.model?dl=1 -O word2vec_with.model
wget https://www.dropbox.com/s/9bk101ih2dy7x4i/word2vec_with.model.syn1neg.npy?dl=1 -O word2vec_with.model.syn1neg.npy
wget https://www.dropbox.com/s/69ln475xwthy8rs/word2vec_with.model.wv.syn0.npy?dl=1 -O word2vec_with.model.wv.syn0.npy
echo "testing"
python3 test.py $1 $2
