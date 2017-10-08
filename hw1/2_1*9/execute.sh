#! /bin/bash
DLAP="$1"
DIR="log"
ANALYSIS="analysis.py"
LOG="$DIR/log"
GRAPH="$DIR/log_graph.png"

rm -rf "$DIR"
mkdir "$DIR"
echo "training"
python3 my_train.py 652
echo "testing"
python3 my_test.py "../test.csv" res.csv
echo "analysing"
python3 "$ANALYSIS"
cat "$LOG"
open "$GRAPH"
