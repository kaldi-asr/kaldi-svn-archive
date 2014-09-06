#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
data=/export/a15/vpanayotov/data

# base url for downloads.
url=www.openslr.org/resources/11

. cmd.sh

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
   /export/a15/vpanayotov/kaldi-egs/ls/s5/data/lm /export/a15/vpanayotov/data/g2p data/local/dict

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

steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train-2kshort data/lang exp/mono || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train-5k data/lang exp/mono exp/mono_ali_5k

# We next train an LDA+MLLT system.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train-5k data/lang exp/mono_ali_5k exp/tri1a || exit 1;


# to be continued. 
