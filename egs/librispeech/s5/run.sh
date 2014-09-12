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

## Optional text corpus normalization and LM training
## These scripts are here primarily as a documentation of the process that has been
## used to build the LM. Most users of this recipe will NOT need/want to run
## this step
#local/lm/train_lm.sh $LM_CORPUS_ROOT \
#  data/lm/norm/tmp data/lm/norm/norm_texts data/lm || exit 1

# the inputs are created by Vassil but we need to include the scripts to create them.
local/prepare_dict.sh --nj 30 --cmd "$train_cmd" \
   /export/a15/vpanayotov/data/lm /export/a15/vpanayotov/data/g2p data/local/dict

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/format_lms.sh /export/a15/vpanayotov/data/lm || exit 1

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
utils/mkgraph.sh --mono data/lang_test_tgsmall exp/mono exp/mono/graph_tgsmall || exit 1
for test in dev-clean dev-other; do
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    exp/mono/graph_tgsmall data/$test exp/mono/decode_tgsmall_$test
done
)&

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train-5k data/lang exp/mono exp/mono_ali_5k

# train a first delta + delta-delta triphone system on a subset of 5000 utterances
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train-5k data/lang exp/mono_ali_5k exp/tri1 || exit 1;

# decode using the tri1 model
(
utils/mkgraph.sh data/lang_test_tgsmall exp/tri1 exp/tri1/graph_tgsmall || exit 1;
for test in dev-clean dev-other; do
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri1/graph_tgsmall data/$test exp/tri1/decode_tgsmall_$test || exit 1;
  steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
     data/$test exp/tri1/decode_{tgsmall,tgmed}_$test  || exit 1;
 done
)&

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train-10k data/lang exp/tri1 exp/tri1_ali_10k || exit 1;


# train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train-10k data/lang exp/tri1_ali_10k exp/tri2b || exit 1;

# decode using the LDA+MLLT model
(
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri2b exp/tri2b/graph_tgsmall || exit 1;
  for test in dev-clean dev-other; do
    steps/decode.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri2b/graph_tgsmall data/$test exp/tri2b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri2b/decode_{tgsmall,tgmed}_$test  || exit 1;
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
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri3b exp/tri3b/graph_tgsmall || exit 1;
  for test in dev-clean dev-other; do
    steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
      exp/tri3b/graph_tgsmall data/$test exp/tri3b/decode_tgsmall_$test || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test  || exit 1;
  done
)&

# align the entire train-clean-100 subset using the tri3b model
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train-clean-100 data/lang exp/tri3b exp/tri3b_ali_clean_100 || exit 1;

# train another LDA+MLLT+SAT system on the entire 100 hour subset
steps/train_sat.sh  --cmd "$train_cmd" \
  4200 40000 data/train-clean-100 data/lang exp/tri3b_ali_clean_100 exp/tri4b || exit 1;

# decode using the tri4b model
(
utils/mkgraph.sh data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall || exit 1;
for test in dev-clean dev-other; do
  steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri4b/graph_tgsmall data/$test exp/tri4b/decode_tgsmall_$test || exit 1;
done
)&

# align train-clean-100 using the tri4b model
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train-clean-100 data/lang exp/tri4b exp/tri4b_ali_clean_100 || exit 1;

# train and test NN model(s) on the 100 hour subset
local/run_nnet2_clean_100.sh || exit 1

# now add the "clean-360" subset to the mix ...
local/data_prep.sh $data/LibriSpeech/train-clean-360 data/train-clean-360 || exit 1
steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train-clean-360 \
  exp/make_mfcc/train-clean-360 $mfccdir || exit 1
steps/compute_cmvn_stats.sh data/train-clean-360 exp/make_mfcc/train-clean-360 $mfccdir || exit 1

# ... and then combine the two sets into a 460 hour one
utils/combine_data.sh data/train-clean-460 data/train-clean-100 data/train-clean-360 || exit 1

# align the new, combined set, using the tri4b model
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train-clean-460 data/lang exp/tri4b exp/tri4b_ali_clean_460 || exit 1;

# train a NN model on the 460 hour set
local/run_nnet2_clean_460.sh || exit 1

# prepare the 500 hour subset
local/data_prep.sh $data/LibriSpeech/train-other-500 data/train-other-500 || exit 1
steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/train-other-500 \
  exp/make_mfcc/train-other-500 $mfccdir || exit 1
steps/compute_cmvn_stats.sh data/train-other-500 exp/make_mfcc/train-other-500 $mfccdir || exit 1

# combine all the data
utils/combine_data.sh data/train-960 data/train-clean-460 data/train-other-500 || exit 1

# now take a decent-sized subset of the combined set
utils/subset_data_dir.sh data/train-960 65000 data/train-65k || exit 1

steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train-65k data/lang exp/tri4b exp/tri4b_ali_65k || exit 1;

# train a SAT model on the 65k mixed (clean + other) subset
steps/train_sat.sh  --cmd "$train_cmd" \
  5300 54000 data/train-65k data/lang exp/tri4b_ali_65k exp/tri4c || exit 1;

# align the entire dataset
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
  data/train-960 data/lang exp/tri4c exp/tri4c_ali_960 || exit 1;

# train NN models on the entire dataset
local/run_nnet2_960.sh || exit 1

# train models on cleaned-up data
local/run_data_cleaning.sh
