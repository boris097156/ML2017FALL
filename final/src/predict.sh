#! /bin/bash
# Usage:
# cd to src/
# ./predict.sh [path of `training_data/`] [path of `testing_data.csv`] [path of `submission.csv`]

TRAIN_DIR=$1
TEST_CSV=$2
OUTPUT_CSV=$3

if [ ! -d "./opencc" ]; then
  git clone https://github.com/yichen0831/opencc-python.git opencc
fi

SEG_TEST=`python3 seg_data.py --train_dir $TRAIN_DIR --test_path $TEST_CSV --output_dir ../data/SC_seg_data`
python3 ensemble_predict.py --load_word2vec1 ../model/SC_minlen1_concat7_str1_size64_window7_min2_sg\
							--load_word2vec2 ../model/SC_minlen2_concat7_str1_size64_window7_min2_sg\
							--load_word2vec3 ../model/SC_minlen1_concat7_str2_size64_window7_min2_sg\
							--word_dim 64 \
							--model_num 3 \
							--seg_test $SEG_TEST\
							--output_csv $OUTPUT_CSV