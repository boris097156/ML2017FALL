#! /bin/bash
echo "training"
python3 logistic.py $3 $4
echo "testing"
python3 "test.py" $5 $6

