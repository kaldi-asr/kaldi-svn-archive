#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
data=/export/a15/vpanayotov/data

# base url for downloads.
url=www.openslr.org/resources/11

. cmd.sh
. path.sh

# you might not want to do this for interactive shells.
set -e

# download the data.  Note: we're using the 100 hour setup for
# now; later in the script we'll download more and use it to train neural
# nets.
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  local/download_and_untar.sh $data $url $part
done

# format the data as Kaldi data directories
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  local/data_prep.sh $data/LibriSpeech/$part data/$part  
done

# the inputs are created by Vassil but we need to include the scripts to create them.
local/prepare_dict.sh --nj 30 --cmd "$train_cmd" \
   /export/a15/vpanayotov/data/lm /export/a15/vpanayotov/data/g2p data/local/dict

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/format_data.sh /export/a15/vpanayotov/data/lm || exit 1

mfccdir=mfcc
for part in dev-clean test-clean dev-other test-other train-clean-100; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
  steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done

# Make some small data subsets for early system-build stages.  Note, there are 29k
# utterances in the train-clean-100 directory which has 100 hours of data.
# For the monophone stages we select the shortest utterances, which should make it
# easier to align the data from a flat start.

utils/subset_data_dir.sh --shortest data/train-clean-100 2000 data/train-2kshort
utils/subset_data_dir.sh data/train-clean-100 5000 data/train-5k
utils/subset_data_dir.sh data/train-clean-100 10000 data/train-10k

# train a monophone system
steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
  data/train-2kshort data/lang exp/mono || exit 1;

# decode using the monophone model
(
utils/mkgraph.sh --mono data/lang_test_tgpr exp/mono exp/mono/graph_tgpr || exit 1
for test in dev-clean dev-other; do
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    exp/mono/graph_tgpr data/$test exp/mono/decode_tgpr_$test
done
)&

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train-5k data/lang exp/mono exp/mono_ali_5k

# train a first delta + delta-delta triphone system on a subset of 5000 utterances
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train-5k data/lang exp/mono_ali_5k exp/tri1 || exit 1;

# decode using the tri1 model
(
utils/mkgraph.sh data/lang_test_tgpr exp/tri1 exp/tri1/graph_tgpr || exit 1;
for test in dev-clean dev-other; do
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri1/graph_tgpr data/$test exp/tri1/decode_tgpr_$test || exit 1;
done
)&

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train-10k data/lang exp/tri1 exp/tri1_ali_10k || exit 1;

# train a second slightly larger delta + delta-delta model
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2500 15000 data/train-10k data/lang exp/tri1_ali_10k exp/tri2a || exit 1;

# decode using the tri2a model
(
utils/mkgraph.sh data/lang_test_tgpr exp/tri2a exp/tri2a/graph_tgpr || exit 1;
for test in dev-clean dev-other; do
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri2a/graph_tgpr data/$test exp/tri2a/decode_tgpr_$test || exit 1;
done
)&

# train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train-10k data/lang exp/tri1_ali_10k exp/tri2b || exit 1;

# decode using the LDA+MLLT model
(
utils/mkgraph.sh data/lang_test_tgpr exp/tri2b exp/tri2b/graph_tgpr || exit 1;
for test in dev-clean dev-other; do
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri2b/graph_tgpr data/$test exp/tri2b/decode_tgpr_$test || exit 1;
done
)&

# Align a 10k utts subset using the tri2b model
steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
  --use-graphs true data/train-10k data/lang exp/tri2b exp/tri2b_ali_10k || exit 1;

# Train tri3b, which is LDA+MLLT+SAT on 10k utts
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train-10k data/lang exp/tri2b_ali_10k exp/tri3b || exit 1;

# decode using the tri3b model
(
utils/mkgraph.sh data/lang_test_tgpr exp/tri3b exp/tri3b/graph_tgpr || exit 1;
for test in dev-clean dev-other; do
  steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri3b/graph_tgpr data/$test exp/tri3b/decode_tgpr_$test || exit 1;
done
)&

# align the entire train-clean-100 subset using the tri3b model
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train-clean-100 data/lang exp/tri3b exp/tri3b_ali_100h || exit 1;

# train another LDA+MLLT+SAT system on the entire 100 hour subset
steps/train_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train-clean-100 data/lang exp/tri3b_ali_100h exp/tri4b || exit 1;

# decode using the tri4b model
(
utils/mkgraph.sh data/lang_test_tgpr exp/tri4b exp/tri4b/graph_tgpr || exit 1;
for test in dev-clean dev-other; do
  steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri4b/graph_tgpr data/$test exp/tri4b/decode_tgpr_$test || exit 1;
done
)&

