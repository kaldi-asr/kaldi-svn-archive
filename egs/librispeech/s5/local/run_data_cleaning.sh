#!/bin/bash


# This script shows how you can do data-cleaning, and exclude data that has a
# higher likelihood of being wrongly transcribed.

stage=1
. ./cmd.sh || exit 1;


. utils/parse_options.sh || exit 1;

set -e


if [ $stage -le 1 ]; then
  steps/cleanup/find_bad_utts.sh --nj 100 --cmd "$train_cmd" data/train_960 data/lang \
    exp/tri6b exp/tri6b_cleanup
fi

# TODO: figure out the threshold value
thresh=0.1
if [ $stage -le 2 ]; then
  cat exp/tri6b_cleanup/all_info.txt | awk -v threshold=$thresh '{ errs=$2;ref=$3; if (errs <= threshold*ref) { print $1; } }' > uttlist
  utils/subset_data_dir.sh --utt-list uttlist data/train_960 data/train.thresh$thresh
fi

if [ $stage -le 3 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train.thresh$thresh data/lang exp/tri6b exp/tri6b_ali_$thresh
fi

if [ $stage -le 4 ]; then
  steps/train_sat.sh  --cmd "$train_cmd" \
    7000 150000 data/train_960 data/lang exp/tri6b_ali_$thresh  exp/tri6b_$thresh || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri6b_$thresh exp/tri6b_$thresh/graph_tgsmall || exit 1
  for test in dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 50 --cmd "$decode_cmd" --config conf/decode.config \
      exp/tri6b_$thresh/graph_tgsmall data/$test exp/tri6b_$thresh/decode_tgsmall_$test || exit 1
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri6b_$thresh/decode_{tgsmall,tgmed}_$test  || exit 1;
  done
fi
