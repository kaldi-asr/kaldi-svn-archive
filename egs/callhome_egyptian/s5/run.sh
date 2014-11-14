#!/bin/bash
#
# Johns Hopkins University (Author : Gaurav Kumar, Daniel Povey)
# Recipe for CallHome Egyptian Arabic
# Made to integrate KALDI with JOSHUA for end-to-end ASR and SMT

. ./cmd.sh
. ./path.sh
mfccdir=`pwd`/mfcc

set -e
set -o pipefail

# Specify the location of the speech files, the transcripts and the lexicon
# These are passed off to other scripts in including the one for data and lexicon prep

eca_speech=/export/corpora/LDC/LDC97S45
eca_transcripts=/export/corpora/LDC/LDC97T19
eca_lexicon=/export/corpora/LDC/LDC99L22
supp_speech=/export/corpora/LDC/LDC2002S37
supp_transcripts=/export/corpora/LDC/LDC2002T38/

numLeavesSAT=6000
numGaussSAT=75000
numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=80000

local/init_arz_system_ibm.sh --eca-lexicon $eca_lexicon
# Added c,j, v to the non silences phones manually
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang


# Make sure that you do not use your test and your dev sets to train the LM
# Some form of cross validation is possible where you decode your dev/set based on an 
# LM that is trained on  everything but that that conversation
rm -rf  data/lm/
local/callhome_train_lms.sh train
local/callhome_create_test_lang.sh

#fi

for dataset in dev tune test train ; do
  utils/fix_data_dir.sh data/$dataset

  steps/make_mfcc_pitch.sh --nj 20 --cmd "$train_cmd" data/$dataset exp/make_mfcc/$dataset $mfccdir || exit 1;

  utils/fix_data_dir.sh data/$dataset

  steps/compute_cmvn_stats.sh data/$dataset exp/make_mfcc/$dataset $mfccdir
  utils/fix_data_dir.sh data/$dataset
  utils/validate_data_dir.sh data/$dataset
done
# Again from Dan's recipe : Reduced monophone training data
# Now-- there are 1.6 million utterances, and we want to start the monophone training
# on relatively short utterances (easier to align), but not only the very shortest
# ones (mostly uh-huh).  So take the 100k shortest ones, and then take 10k random
# utterances from those.

utils/subset_data_dir.sh --per-spk data/train 10 data/train_sub
steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_sub data/lang exp/mono0a    

utils/subset_data_dir.sh --per-spk data/train 30 data/train_sub2
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_sub2 data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    1000 10000 data/train_sub2 data/lang exp/mono0a_ali exp/tri1 || exit 1;


(utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
 steps/decode.sh --nj 13 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri1/graph data/tune exp/tri1/decode_tune)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    1400 15000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;             
  steps/decode.sh --nj 13 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri2/graph data/tune exp/tri2/decode_tune || exit 1;
)&


steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   1800 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 13 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/tune exp/tri3a/decode_tune || exit 1;
)&

# Next we'll use fMLLR and train with SAT (i.e. on 
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  2200 25000 data/train data/lang exp/tri3a_ali  exp/tri4a || exit 1;
                                                                                 
(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 13 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/tune exp/tri4a/decode_tune
)&


steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/tri4a exp/tri4a_ali || exit 1;

# Reduce the number of gaussians
steps/train_sat.sh  --cmd "$train_cmd" \
  $numLeavesSAT $numGaussSAT data/train data/lang exp/tri4a_ali  exp/tri5b || exit 1;

utils/mkgraph.sh data/lang_test exp/tri5b exp/tri5b/graph
(
  steps/decode_fmllr_extra.sh --nj 13 --cmd "$decode_cmd" --num-threads 4 --parallel-opts " -pe smp 4" \
    --config conf/decode.config  --scoring-opts "--min-lmwt 8 --max-lmwt 12"\
   exp/tri5b/graph data/tune exp/tri5b/decode_tune
)&

steps/align_fmllr.sh \
  --boost-silence 0.5 --nj 64 --cmd "$train_cmd" \
  data/train data/lang exp/tri5b exp/tri5b_ali

steps/train_ubm.sh \
  --cmd "$train_cmd" 850 \
  data/train data/lang exp/tri5a_ali exp/ubm5

steps/train_sgmm2.sh \
  --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
  data/train data/lang exp/tri5b_ali exp/ubm5/final.ubm exp/sgmm5b

utils/mkgraph.sh data/lang_test exp/sgmm5b exp/sgmm5b/graph

(

  steps/decode_sgmm2.sh --nj 13 --cmd "$decode_cmd" --num-threads 5 --parallel-opts " -pe smp 5" \
    --config conf/decode.config  --scoring-opts "--min-lmwt 8 --max-lmwt 16" --transform-dir exp/tri5b/decode_tune \
   exp/sgmm5b/graph data/tune exp/sgmm5b/decode_tune
)&

steps/align_sgmm2.sh \
  --nj 64  --cmd "$train_cmd" --transform-dir exp/tri5b_ali \
  --use-graphs true --use-gselect true \
  data/train data/lang exp/sgmm5b exp/sgmm5b_ali

steps/make_denlats_sgmm2.sh \
  --nj 64 --sub-split 64 --num-threads 4 --parallel-opts "-pe smp 4"\
  --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5b_ali \
  data/train data/lang exp/sgmm5b_ali exp/sgmm5b_denlats

steps/train_mmi_sgmm2.sh \
  --cmd "$train_cmd" --drop-frames true --transform-dir exp/tri5b_ali --boost 0.1 \
  data/train data/lang exp/sgmm5b_ali exp/sgmm5b_denlats \
  exp/sgmm5b_mmi_b0.1

(
for iter in 1 2 3 4; do                                                            
  decode=exp/sgmm5b_mmi_b0.1/decode_tune_it$iter                                  
  mkdir -p $decode                                                                 
  steps/decode_sgmm2_rescore.sh  \
    --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5b/decode_tune \
    data/lang_test data/tune/  exp/sgmm5b/decode_tune $decode
done
) &


dnn_cpu_parallel_opts=(--minibatch-size 128 --max-change 10 --num-jobs-nnet 8 --num-threads 16 \
                       --parallel-opts "-pe smp 16" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
                       --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")

steps/nnet2/train_pnorm_ensemble.sh \
  --mix-up 5000  --initial-learning-rate 0.008 --final-learning-rate 0.0008\
  --num-hidden-layers 4 --pnorm-input-dim 2000 --pnorm-output-dim 200\
  --cmd "$train_cmd" \
  "${dnn_gpu_parallel_opts[@]}" \
  --ensemble-size 4 --initial-beta 0.1 --final-beta 5 \
  data/train data/lang exp/tri5b_ali exp/tri6a_dnn

(
  steps/nnet2/decode.sh --nj 13 --cmd "$decode_cmd" --num-threads 4 --parallel-opts " -pe smp 4"   \
    --scoring-opts "--min-lmwt 8 --max-lmwt 16" --transform-dir exp/tri5b/decode_tune exp/tri5b/graph data/tune exp/tri6a_dnn/decode_tune
) &
wait
exit 0;
