#!/bin/bash

# Decodes all test sets for a given pretrained checkpoint
# Check the Checkpoint's folder to see the created folders that contain the
# hypotheses and refs.
CKPT="$1"

if [ -z $CKPT ]; then
  echo 'You need to provide a checkpoint .pth file for decoding.'
  exit 1
fi

shift 1

DATA_PATH="./data/how2"
FEAT_PATH="./data/how2/features"

BS=${BS:-8}
NAME="${CKPT/.pth/}_beam${BS}/"

# Decode test_2016_flickr first
python3 train.py --beam_size $BS --exp_name $NAME --exp_id "${TEST_SET}" --dump_path . \
  --reload_model "${CKPT},${CKPT}" --data_path "${DATA_PATH}" \
  --encoder_only false --lgs 'pt-en' --mmt_step "pt-en" --emb_dim 512 --n_layers 6 --n_heads 8 \
  --dropout '0.4' --attention_dropout '0.1' --gelu_activation true \
  --batch_size 16 --optimizer "adam,lr=0.0001" \
    --eval_bleu true --eval_only true --reg_enc_bias false --num_of_regions 36 \
    --region_feats_path $FEAT_PATH --image_names $DATA_PATH $@
