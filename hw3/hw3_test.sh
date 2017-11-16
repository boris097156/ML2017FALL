#! /bin/bash
echo "downloading models"
wget https://www.dropbox.com/s/lf5qx5oibus6by8/model0.h5?dl=1 -O model0.h5
wget https://www.dropbox.com/s/bc2wohtngmdanjl/model2.h5?dl=1 -O model2.h5
echo "testing"
python3 hw3_ensemble.py $1 $2
