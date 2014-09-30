#!/bin/bash
#
# Copyright 2014 Jan Trmal, Hainan Xu, Xiaohui Zhang, Xin Lei, Minhua Wu
# Apache 2.0
#
# This is a shell script to run BOLT evaluation recipe. It's recommended
# that you run the commands one by one by copy-and-paste into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so
# you should run this on a machine that has sufficient memory.

set -e
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./conf.sh

# Data Preparation, 
local/callhome_data_prep.sh "$CALLHOME_MA_CORPUS_A" "$CALLHOME_MA_CORPUS_T" data/local || exit 1;
local/hkust_data_prep.sh --corpus-id "hkust" "$HKUST_MA_CORPUS_A" "$HKUST_MA_CORPUS_T" data/local || exit 1;
local/hub5_data_prep.sh --corpus-id "hub5" "$HUB5_MA_CORPUS_A" "$HUB5_MA_CORPUS_T" data/local || exit 1;
local/rt04f_data_prep.sh --corpus-id "rt04f"  "$RT04F_MA_TRAIN_CORPUS_A" "$RT04F_MA_TRAIN_CORPUS_T" \
  "$RT04F_MA_DEV_CORPUS_A" "$RT04F_MA_DEV_CORPUS_T" data/local || exit 1;

# Lexicon Preparation,
local/callhome_prepare_dict.sh || exit 1;

# Phone Sets, questions, L compilation 
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

# LM training
local/bolt_train_lms_interpolate.sh

# G compilation, check LG composition
local/bolt_format_data.sh
# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=param
for x in train dev; do 
  steps/make_mfcc_pitch.sh --nj $train_nj --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done
# after this, the next command will remove the small number of utterances
# that couldn't be extracted for some reason (e.g. too short; no such file).
utils/fix_data_dir.sh data/train || exit 1;
utils/fix_data_dir.sh data/dev || exit 1;

#Subset the train dir as the first stages do not 
#really need some huge amount of training data
local/subset_train_dirs.sh

steps/train_mono.sh --nj 4 --cmd "$train_cmd"\
  data/train_sub1 data/lang exp/mono0a || exit 1;


# Monophone decoding.
utils/mkgraph.sh --mono data/lang_test exp/mono0a exp/mono0a/graph || exit 1
# note: local/decode.sh calls the command line once for each
# test, and afterwards averages the WERs into (in this case
# exp/mono/decode/

steps/decode.sh --config conf/decode.config --nj $decode_nj "${extra_decoding_opts[@]}"\
  --cmd "$decode_cmd" --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
  exp/mono0a/graph data/dev exp/mono0a/decode_dev


# Get alignments from monophone system.
steps/align_si.sh --nj $train_nj --cmd "$train_cmd"\
  data/train_sub2 data/lang exp/mono0a exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd"\
 2500 20000 data/train_sub2 data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;

steps/decode.sh --config conf/decode.config --nj $decode_nj "${extra_decoding_opts[@]}"\
  --cmd "$decode_cmd" --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
  exp/tri1/graph data/dev exp/tri1/decode_dev


# align tri1
steps/align_si.sh --nj $train_nj --cmd "$train_cmd"\
  data/train_sub3 data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2 [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd"\
 2500 20000 data/train_sub3 data/lang exp/tri1_ali exp/tri2 || exit 1;

# decode tri2
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode.sh --config conf/decode.config --nj $decode_nj "${extra_decoding_opts[@]}"\
  --cmd "$decode_cmd" --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
  exp/tri2/graph data/dev exp/tri2/decode_dev

# train and decode tri2b [LDA+MLLT]

steps/align_si.sh --nj $train_nj --cmd "$train_cmd"\
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, 
steps/train_lda_mllt.sh --cmd "$train_cmd"\
 2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
steps/decode.sh --nj $decode_nj --config conf/decode.config "${extra_decoding_opts[@]}"\
  --cmd "$decode_cmd" --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
  exp/tri3a/graph data/dev exp/tri3a/decode_dev
# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd"\
  data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd"\
  2500 20000 data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
steps/decode_fmllr.sh --nj $decode_nj --config conf/decode.config "${extra_decoding_opts[@]}"\
  --cmd "$decode_cmd" --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
  exp/tri4a/graph data/dev exp/tri4a/decode_dev

steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd"\
  data/train data/lang exp/tri4a exp/tri4a_ali

# Building a larger SAT system.

steps/train_sat.sh --cmd "$train_cmd"\
  3500 160000 data/train data/lang exp/tri4a_ali exp/tri5a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
steps/decode_fmllr_extra.sh --nj $decode_nj --cmd "$decode_cmd" "${extra_decoding_opts[@]}"\
   --config conf/decode.config  --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
   exp/tri5a/graph data/dev exp/tri5a/decode_dev || exit 1;

./local/run_dnn.sh
./local/run_dnn_mpe.sh
exit 0;

# MMI starting from system in tri5a
# Later we'll use all of it.
steps/align_fmllr.sh --nj $decode_nj --cmd "$train_cmd"\
  data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;

steps/make_denlats.sh --nj $train_nj --cmd "$train_cmd" "${extra_decoding_opts[@]}"\
  --config conf/decode.config --transform-dir exp/tri5a_ali \
  data/train data/lang exp/tri5a exp/tri5a_denlats || exit 1;

steps/train_mmi.sh --drop-frames true --num-iters 4 --boost 0.1 --cmd "$train_cmd"\
  data/train data/lang exp/tri5a_ali exp/tri5a_denlats exp/tri5a_mmi_b0.1 || exit 1;

for iter in `seq 1 4` ; do
  steps/decode.sh --iter $iter --nj $decode_nj --cmd "$train_cmd" "${extra_decoding_opts[@]}" \
    --transform-dir exp/tri5a/decode_dev  --config conf/decode.config \
    exp/tri5a/graph data/dev exp/tri5a_mmi_b0.1/decode_dev.it$iter  || exit 1 ; 
done

# Do MPE.
steps/train_mpe.sh --num-iters 4 --boost 0.1  --cmd "$train_cmd" \
  data/train data/lang exp/tri5a_ali exp/tri5a_denlats exp/tri5a_mpe_b0.1 || exit 1;

for iter in `seq 1 4` ; do
  steps/decode.sh --iter $iter --nj $decode_nj --cmd "$train_cmd" "${extra_decoding_opts[@]}" \
    --transform-dir exp/tri5a/decode_dev  --config conf/decode.config \
    exp/tri5a/graph data/dev exp/tri5a_mpe_b0.1/decode_dev.it$iter  || exit 1 ; 
done


# getting results (see RESULTS file)
for x in exp/*/decode; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

exit 0;
#steps/train_mce.sh (Minimum Classification Error) scripts do not longer exist.
#Not sure why, though..

# Do MCE.
#steps/train_mce.sh --cmd "$train_cmd" data/train data/lang exp/tri5a_ali exp/tri5a_denlats exp/tri5a_mce || exit 1;
#
#steps/decode.sh --nj 16 --config conf/decode.config \
#  --transform-dir exp/tri5a/decode --cmd "$train_cmd" \
#  exp/tri5a/graph data/dev exp/tri5a_mce/decode || exit 1 ;


