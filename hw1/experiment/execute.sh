#! /bin/bash
OUTPUT_DIR="$1"
LOG_DIR="$OUTPUT_DIR/log"
LOG="$LOG_DIR/log"
GRAPH="$LOG_DIR/log_graph.png"
OUTPUT="$OUTPUT_DIR/res.csv"

rm -rf "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR"
mkdir "$LOG_DIR"
echo "training"
python3 my_train.py 652
echo "testing"
python3 my_test.py "../test.csv" "$OUTPUT"
echo "analysing"
python3 analysis.py "$OUTPUT_DIR"
cat "$LOG"
open "$GRAPH"
