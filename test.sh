#!/bin/bash

# Get .sh file path
ROOT_DIR=$(dirname $(readlink -f $0))

BIN_DIR=$ROOT_DIR/src
DATA_DIR=$ROOT_DIR/data
MODEL_DIR=$ROOT_DIR/saved_model

python3 $BIN_DIR/test.py \
  --test_data $DATA_DIR/nikluge-sa-2022-test.jsonl \
  --base_model xlm-roberta-base \
  --entity_property_model_path $MODEL_DIR/category_extraction/epoch_20.pt \
  --polarity_model_path $MODEL_DIR/polarity_classification/epoch_20.pt \
  --batch_size 8 \
  --max_len 256 \
  && echo "Testing is done"