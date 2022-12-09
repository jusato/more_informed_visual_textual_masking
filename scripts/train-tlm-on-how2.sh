#!/bin/bash

# Trains TLM on How2 for quick / demonstrational purposes

DATA_PATH="./data/how2"
DUMP_PATH="./models"
EXP_NAME="tlm-on-how2"
EPOCH_SIZE=29000

python3 train.py --exp_name $EXP_NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
  --lgs 'pt-en' --clm_steps '' --mlm_steps 'pt-en' \
  --emb_dim 512 --n_layers 6 --n_heads 8 --dropout '0.1' \
  --attention_dropout '0.1' --gelu_activation true --batch_size 16 --bptt 1 \
  --optimizer 'adam,lr=0.0001' --epoch_size ${EPOCH_SIZE} --max_epoch 100000 \
  --validation_metrics '_valid_pt_en_mlm_ppl' --stopping_criterion '_valid_pt_en_mlm_ppl,50' \
  --fp16 false --save_periodic 5 --iter_seed 12345 --other_seed 12345 $@
