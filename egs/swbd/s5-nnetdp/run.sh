#!/bin/bash

# Note:  this is a work in progress, but should run up to the point where it says:
# I AM HERE 

exit 1;
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.

. cmd.sh

# Data prep

#local/swbd_p1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2
local/swbd_p1_data_prep.sh /export/corpora3/LDC/LDC97S62 

local/swbd_p1_prepare_dict.sh

utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

local/swbd_p1_train_lms.sh

local/swbd_p1_format_data.sh

# Data preparation and formatting for eval2000 (note: the "text" file
# is not very much preprocessed; for actual WER reporting we'll use
# sclite.
#local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/2000_hub5_eng_eval_tr
local/eval2000_data_prep.sh /export/corpora2/LDC/LDC2002S09/hub5e_00 /export/corpora2/LDC/LDC2002T43

. cmd.sh
# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfcc

steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/train exp/make_mfcc/train $mfccdir 
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
# after this, the next command will remove the small number of utterances
# that couldn't be extracted for some reason (e.g. too short; no such file).
utils/fix_data_dir.sh data/train

steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_mfcc/eval2000 $mfccdir 
steps/compute_cmvn_stats.sh data/eval2000 exp/make_mfcc/eval2000 $mfccdir
utils/fix_data_dir.sh data/eval2000 # remove segments that had problems, e.g. too short.

# Now make the filter-bank features.
for x in train eval2000; do
  mkdir -p data-fbank
  cp -r data/$x data-fbank/$x
  steps/make_fbank.sh --nj 20 --cmd "$train_cmd" data-fbank/$x exp/make_fbank/$x $featdir 
  steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $featdir 
done


# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
# LM training data.   However, they will be in the lexicon, plus speakers
# may overlap, so it's still not quite equivalent to a test set.

for data in data data-fbank; do
  utils/subset_data_dir.sh --first $data/train 4000 $data/train_dev 
  n=$[`cat $data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $data/train $n $data/train_nodev
done



# Now-- there are 264k utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.

for data in data data-fbank; do
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh  $data/train_100kshort 10000 $data/train_10k
  local/remove_dup_utts.sh 100 $data/train_10k $data/train_10k_nodup

  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --first $data/train_nodev 30000 $data/train_30k
  local/remove_dup_utts.sh 200 $data/train_30k $data/train_30k_nodup

  local/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_nodev 100000 $data/train_100k
  local/remove_dup_utts.sh 200 $data/train_100k $data/train_100k_nodup
done


steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_10k_nodup data/lang exp/mono0a 

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup data/lang exp/mono0a_ali exp/tri1

utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph

steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
  exp/tri1/graph data/eval2000 exp/tri1/decode_eval2000

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_30k_nodup data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_30k_nodup data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/eval2000 exp/tri2/decode_eval2000 || exit 1;
)&


steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 30k_nodup data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 20000 data/train_30k_nodup data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/eval2000 exp/tri3a/decode_eval2000 || exit 1;
)&

# Align all data with LDA+MLLT system (tri3a)
steps/align_si.sh --nj 30 --cmd "$train_cmd" --use-graphs true \
   data/train_30k_nodup data/lang exp/tri3a exp/tri3a_ali


(
  ## Build a LDA+MLLT system just for decorrelation, on the raw filterbank
  ## features, but using the tree from the tri2b system. This is just to get the
  ## decorrelating transform and will also allow us to get SAT on the
  ## filterbank features.
  featdim=`feat-to-dim scp:data-fbank/train/feats.scp -` 

  local/train_lda_mllt_notree.sh --cmd "$train_cmd" --dim $featdim \
    --splice-opts "--left-context=0 --right-context=0" \
    --realign-iters "" 20000 data-fbank/train_30k_nodup data/lang exp/tri3a_ali exp/tri4b

  # We decode this out of curiosity; it won't be very good.
  steps/decode.sh --nj 20 --config conf/decode.config --cmd "$decode_cmd" \
    exp/tri3a/graph data-fbank/eval2000 exp/tri4b/decode_eval2000 &

  ## Train fMLLR on top of fbank features.  Note: we give it directory
  ## tri4b instead of an alignment directory like tri4b_ali, because we want
  ## to use the alignments in tri4b which were just copied from the conventional
  ## system.
  local/train_sat_notree.sh --cmd "$train_cmd" \
    --realign-iters "" 20000 data-fbank/train_30k_nodup data/lang exp/tri4b exp/tri5b

  ## Decode the test data with this system (will need it for nnet testing,
  ## for the transforms.)
  ## Use transcripts from the 3a system which had frame splicing -> better 
  ## supervision.
  steps/decode_fmllr.sh --si-dir exp/tri3a/decode_eval2000 \
    --nj 30 --config conf/decode.config --cmd "$decode_cmd" \
    exp/tri3a/graph data-fbank/eval2000 exp/tri5b/decode_eval2000
)

# Do SAT on top of the standard splice+LDA+MLLT system,
# because we want good alignments to build the 2-level tree on top of,
# for the neural nset system.
steps/train_sat.sh 2500 20000 data/train_30k_nodup data/lang exp/tri3a_ali exp/tri4a

# Align 30k_nodup data with LDA+MLLT+SAT system (tri4a)
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" --use-graphs true \
  data/train_30k_nodup data/lang exp/tri4a exp/tri4a_ali || exit 1;

## First we just build a two-level tree, using the LDA+MLLT+SAT system's
## alignments and features, since that system is pretty good.
local/train_two_level_tree.sh 200 5000 data/train_30k_nodup data/lang exp/tri4a_ali exp/tri5a_tree

local/train_nnet1.sh --num-iters 7 --max-iter-inc 4 \
 10000 data-fbank/train_30k_nodup data/lang exp/tri5b exp/tri5a_tree exp/tri6a_nnet

utils/mkgraph.sh data/lang_test exp/tri6a_nnet exp/tri6a_nnet/graph

iter=7
local/decode_nnet1.sh --transform-dir exp/tri5b/decode_eval2000 \
  --iter $iter --config conf/decode.config --nj 30 \
  --cmd "$decode_cmd" exp/tri6a_nnet/graph data-fbank/eval2000 exp/tri6a_nnet/decode_eval2000_it$iter

local/train_nnet1.sh --num-iters 7 --max-iter-inc 4 --hidden-layer-size 1000 \
 10000 data-fbank/train_30k_nodup data/lang exp/tri5b exp/tri5a_tree exp/tri6b_nnet

iter=7
local/decode_nnet1.sh --transform-dir exp/tri5b/decode_eval2000 \
  --iter $iter --config conf/decode.config --nj 30 \
  --cmd "$decode_cmd" exp/tri6a_nnet/graph data-fbank/eval2000 exp/tri6b_nnet/decode_eval2000_it$iter

local/train_nnet1.sh --num-iters 7 --max-iter-inc 4 --hidden-layer-size 1000 \
  --initial-layer-context "4,4" \
  10000 data-fbank/train_30k_nodup data/lang exp/tri5b exp/tri5a_tree exp/tri6c_nnet

iter=7
local/decode_nnet1.sh --transform-dir exp/tri5b/decode_eval2000 \
  --iter $iter --config conf/decode.config --nj 30 \
  --cmd "$decode_cmd" exp/tri6a_nnet/graph data-fbank/eval2000 exp/tri6c_nnet/decode_eval2000_it$iter

local/train_nnet1.sh --num-iters 9 --max-iter-inc 6 --hidden-layer-size 1000 \
  --add-layer-iters "3" --initial-layer-context "4,4" \
  10000 data-fbank/train_30k_nodup data/lang exp/tri5b exp/tri5a_tree exp/tri6d_nnet

iter=7
local/decode_nnet1.sh --transform-dir exp/tri5b/decode_eval2000 \
  --iter $iter --config conf/decode.config --nj 30 \
  --cmd "$decode_cmd" exp/tri6a_nnet/graph data-fbank/eval2000 exp/tri6d_nnet/decode_eval2000_it$iter


exit 0;
## From here is junk.


# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri3a exp/tri3a_ali_100k_nodup || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  2500 20000 data/train_100k_nodup data/lang exp/tri3a_ali_100k_nodup exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/eval2000 exp/tri4a/decode_eval2000
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/train_dev exp/tri4a/decode_train_dev
)&

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri4a exp/tri4a_ali_100k_nodup

local/run_sgmm.sh
#local/run_sgmm2.sh

# Building a larger SAT system.

steps/train_sat.sh --cmd "$train_cmd" \
  3500 100000 data/train_100k_nodup data/lang exp/tri4a_ali_100k_nodup exp/tri5a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
  steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config \
   --nj 30 exp/tri5a/graph data/eval2000 exp/tri5a/decode_eval2000 || exit 1;
)

# MMI starting from system in tri5a.  Use the same data (100k_nodup).
# Later we'll use all of it.
steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
  data/train_100k_nodup data/lang exp/tri5a exp/tri5a_ali_100k_nodup || exit 1;
steps/make_denlats.sh --nj 40 --cmd "$decode_cmd" --transform-dir exp/tri5a_ali_100k_nodup \
   --config conf/decode.config \
   --sub-split 50 data/train_100k_nodup data/lang exp/tri5a exp/tri5a_denlats_100k_nodup  || exit 1;
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 \
  data/train_100k_nodup data/lang exp/tri5a_{ali,denlats}_100k_nodup exp/tri5a_mmi_b0.1 || exit 1;

steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri5a/decode_eval2000 \
  exp/tri5a/graph data/eval2000 exp/tri5a_mmi_b0.1/decode_eval2000 &

steps/train_diag_ubm.sh --silence-weight 0.5 --nj 40 --cmd "$train_cmd" \
  700 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm

steps/train_mmi_fmmi.sh --learning-rate 0.005 \
  --boost 0.1 --cmd "$train_cmd" \
 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm exp/tri5a_denlats_100k_nodup \
   exp/tri5a_fmmi_b0.1 || exit 1;
 for iter in 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
     --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
     exp/tri5a/graph data/eval2000 exp/tri5a_fmmi_b0.1/decode_eval2000_it$iter &
 done

# Recipe with indirect differential [doesn't make difference here]
steps/train_mmi_fmmi_indirect.sh \
  --boost 0.1 --cmd "$train_cmd" \
 data/train_100k_nodup data/lang exp/tri5a_ali_100k_nodup exp/tri5a_dubm exp/tri5a_denlats_100k_nodup \
   exp/tri5a_fmmi_b0.1_indirect || exit 1;

 for iter in 4 5 6 7 8; do
  steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
     --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
     exp/tri5a/graph data/eval2000 exp/tri5a_fmmi_b0.1_indirect/decode_eval2000_it$iter &
 done

# Note: we haven't yet run with all the data.


# getting results (see RESULTS file)
for x in exp/*/decode*; do [ -d $x ] && grep Sum $x/score_*/*.sys | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null



