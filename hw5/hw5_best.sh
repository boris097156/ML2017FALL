#! /bin/bash
echo "downloading model"
wget https://www.dropbox.com/s/b96ihhisa7x8j9z/best.h5?dl=1 -O best.h5
echo 'hw5_best'
python3 test.py $1 $2
