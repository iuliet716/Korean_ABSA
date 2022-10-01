#!/bin/bash

# Get .sh file path
ROOT_DIR=$(dirname $(readlink -f $0))

BIN_DIR=$ROOT_DIR/src
DATA_DIR=$ROOT_DIR/data
MODEL_DIR=$ROOT_DIR/saved_model

MLFLOW_EXPERIMENT_NAME=$1
MLFLOW_RUN_NAME=$2

python3 $BIN_DIR/train.py \
  --experiment_name $MLFLOW_EXPERIMENT_NAME \
  --run_name $MLFLOW_RUN_NAME \
  --train_data $DATA_DIR/nikluge-sa-2022-train.jsonl \
  --dev_data $DATA_DIR/nikluge-sa-2022-dev.jsonl \
  --base_model xlm-roberta-base \
  --do_eval \
  --learning_rate 3e-6 \
  --eps 1e-8 \
  --num_train_epochs 20 \
  --entity_property_model_path $MODEL_DIR/category_extraction \
  --polarity_model_path $MODEL_DIR/polarity_classification \
  --batch_size 8 \
  --max_len 256 \
  && echo "Training is done"