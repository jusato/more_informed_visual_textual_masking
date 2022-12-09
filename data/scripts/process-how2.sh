#!/bin/bash

# Export moses path
PATH=mosesdecoder/scripts/tokenizer:${PATH}

REPLACE_UNICODE_PUNCT=replace-unicode-punctuation.perl
NORM_PUNC=normalize-punctuation.perl
REM_NON_PRINT_CHAR=remove-non-printing-char.perl
TOKENIZER=tokenizer.perl
LOWERCASE=lowercase.perl

ROOT=`dirname $0`
ROOT=`realpath $ROOT`
LOWER_REMOVE_ACCENT="${ROOT}/lowercase_and_remove_accent.py"
BINARIZE="${ROOT}/../../preprocess.py"


DATA_PATH="how2"
RAW_PATH="how2/raw"

for lg in en pt; do
  for split in train val test; do
    if [ $split == "val" ]; then
      osplit="valid"
    else
      osplit=${split}
    fi

    file="${RAW_PATH}/${split}.${lg}"
    if [ -f ${file}.gz ]; then
      zcat "${file}.gz" | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | \
        $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads 2 -l $lg | $LOWER_REMOVE_ACCENT | \
        fastBPE/fast applybpe_stream bpe/bpe50k.codes bpe/bpe50k.vocab > ${DATA_PATH}/${osplit}.en-pt.${lg} &
    fi
  done
done
wait

# binarize
pushd ${DATA_PATH}
rm *.pth
for file in *.en-pt.{pt,en}; do
  python3 $BINARIZE ../bpe/bpe50k.vocab $file
done

# Create test links
# for f in test_2016_flickr*; do
#   ln -s ${f} ${f/_2016_flickr/}
# done

ln -s val.order valid.order
