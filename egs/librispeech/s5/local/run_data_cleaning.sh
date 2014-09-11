#!/bin/bash


# This script shows how you can do data-cleaning, and exclude data that has a
# higher likelihood of being wrongly transcribed.

. cmd.sh
. path.sh
set -e

steps/cleanup/find_bad_utts.sh --nj 100 --cmd "$train_cmd" data/train-960 data/lang \
  exp/tri4c exp/tri4c_cleanup

# TODO: figure out the threshold value
thresh=0.1
cat exp/tri4c_cleanup/all_info.txt | awk -v threshold=$thresh '{ errs=$2;ref=$3; if (errs <= threshold*ref) { print $1; } }' > uttlist
utils/subset_data_dir.sh --utt-list uttlist data/train-960 data/train.thresh$thresh

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train.thresh$thresh data/lang exp/tri4c exp/tri4c_ali_$thresh

steps/train_sat.sh  --cmd "$train_cmd" \
  7500 210000 data/train-960 data/lang exp/tri4c_ali_$thresh  exp/tri4c_$thresh || exit 1;

utils/mkgraph.sh data/lang_test_tgpr exp/tri4c_$thresh exp/tri4c_$thresh/graph || exit 1
for test in dev-clean dev-other; do
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    exp/tri4c_$thresh/graph data/$test exp/tri4c_$thresh/decode_$test || exit 1
done
