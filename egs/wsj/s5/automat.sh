#!/bin/bash 

. path.sh
. cmd.sh

kws_run.sh \
  conversational.dev \
  cant \
  /mnt/matylda6/ihannema/kaldi/kaldi/trunk/egs/bp101/exp/tri3b_but69/decode_conversational.dev_25_10_8 \
  true \
  25 \
  1 \
  1 \
  true \
  results_ndeterminize || exit 1
