#!/bin/bash

# Trains an NMT from scratch on How2

DATA_PATH="./data/how2"
DUMP_PATH="./models"
EXP_NAME="nmt-from-scratch-how2"
EPOCH_SIZE=29000

python3 train.py --beam_size 8 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'pt-en' --mt_step "pt-en" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 16 --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,warmup_updates=4000" \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 \
  --stopping_criterion 'valid_pt-en_mt_bleu,20' --validation_metrics 'valid_pt-en_mt_bleu' \
  --iter_seed 12345 --other_seed 12345 $@
