#! /bin/bash
echo "簡體中文"
TEST_DATA=$1
OUTPUT=$2
python3 ensemble_predict.py --load_word2vec1 ../model/SC_minlen1_concat7_str1_size64_window7_min2_alpha0.025_iter5_sg_0.549 \
							--load_word2vec2 ../model/SC_minlen2_concat7_str1_size64_window7_min2_alpha0.025_iter5_sg_0.549 \
							--load_word2vec3 ../model/SC_minlen1_concat7_str2_size64_window7_min2_alpha0.025_iter5_sg_0.551 \
							--word_dim 64 \
							--model_num 3 \
							--seg_test $TEST_DATA\
							--output_csv $OUTPUT