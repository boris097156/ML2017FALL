#! /bin/bash
echo "preprocessing"
python3 word2vec.py $1 $2
echo "training"
python3 train.py $1
