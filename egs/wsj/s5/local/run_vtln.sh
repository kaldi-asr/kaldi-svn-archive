#!/bin/bash

. cmd.sh
featdir=mfcc

# train linear vtln
steps/train_lvtln.sh --cmd "$train_cmd" 2500 15000 \
  data/train_si84 data/lang exp/tri2a exp/tri3d

cp -rT data/train_si84 data/train_si84_vtln
cp exp/tri3d/final.warp data/train_si84_vtln/spk2warp
steps/make_mfcc.sh --nj 20 --cmd "run.pl" data/train_si84_vtln exp/make_mfcc/train_si84_vtln $featdir 
steps/compute_cmvn_stats.sh data/train_si84_vtln exp/make_mfcc/train_si84_vtln $featdir  

utils/mkgraph.sh data/lang_test_tgpr exp/tri3d exp/tri3d/graph_tgpr

steps/decode_lvtln.sh  --nj 10 --cmd "$decode_cmd" \
  exp/tri3d/graph_tgpr data/test_dev93 exp/tri3d/decode_tgpr_dev93 &
steps/decode_lvtln.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3d/graph_tgpr data/test_eval92 exp/tri3d/decode_tgpr_eval92 &

for test in dev93 eval92; do
 cp -rT data/test_$test data/test_${test}_vtln
 cp exp/tri3d/decode_tgpr_${test}/final.warp data/test_${test}_vtln/spk2warp
 steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/test_${test}_vtln exp/make_mfcc/test_${test}_vtln $featdir  
 steps/compute_cmvn_stats.sh data/test_${test}_vtln exp/make_mfcc/test_${test}_vtln $featdir  
done


(
 set -e
 steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  2500 15000 data/train_si84_vtln data/lang exp/tri3d exp/tri4d

 utils/mkgraph.sh data/lang_test_tgpr exp/tri4d exp/tri4d/graph_tgpr

 steps/decode.sh --nj 10 --cmd "$decode_cmd" \
    exp/tri4d/graph_tgpr data/test_dev93_vtln exp/tri4d/decode_tgpr_dev93
 steps/decode.sh --nj 8 --cmd "$decode_cmd" \
    exp/tri4d/graph_tgpr data/test_eval92_vtln exp/tri4d/decode_tgpr_eval92

 steps/train_sat.sh 2500 15000 data/train_si84_vtln data/lang exp/tri4d exp/tri5d
 utils/mkgraph.sh data/lang_test_tgpr exp/tri5d exp/tri5d/graph_tgpr

 steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
   exp/tri5d/graph_tgpr data/test_dev93_vtln exp/tri5d/decode_tgpr_dev93
 steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
   exp/tri5d/graph_tgpr data/test_eval92_vtln exp/tri5d/decode_tgpr_eval92
)

# Baseline with no VTLN:
#%WER 2.06 [ 258 / 12533, 37 ins, 47 del, 174 sub ] exp/tri3b/decode/wer_4
#%WER 10.17 [ 1275 / 12533, 123 ins, 191 del, 961 sub ] exp/tri3b/decode_ug/wer_13

# With VTLN:
#%WER 1.99 [ 250 / 12533, 18 ins, 70 del, 162 sub ] exp/tri5d/decode/wer_10
#%WER 9.89 [ 1239 / 12533, 119 ins, 203 del, 917 sub ] exp/tri5d/decode_ug/wer_13
