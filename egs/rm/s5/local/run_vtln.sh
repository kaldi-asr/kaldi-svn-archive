#!/bin/bash

. cmd.sh
featdir=mfcc

# train linear vtln
steps/train_lvtln.sh --cmd "$train_cmd" 1800 9000 \
  data/train data/lang exp/tri2a exp/tri3d
cp -rT data/train data/train_vtln
cp exp/tri3d/final.warp data/train_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/train_vtln exp/make_mfcc/train_vtln $featdir  
steps/compute_cmvn_stats.sh data/train_vtln exp/make_mfcc/train_vtln $featdir  
 utils/mkgraph.sh data/lang exp/tri3d exp/tri3d/graph
steps/decode_lvtln.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3d/graph data/test exp/tri3d/decode

cp -rT data/test data/test_vtln
cp exp/tri3d/decode/final.warp data/test_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/test_vtln exp/make_mfcc/test_vtln $featdir  
steps/compute_cmvn_stats.sh data/test_vtln exp/make_mfcc/test_vtln $featdir  

(
 steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  1800 9000 data/train_vtln data/lang exp/tri3d exp/tri4d
 utils/mkgraph.sh data/lang exp/tri4d exp/tri4d/graph

 steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
    exp/tri4d/graph data/test_vtln exp/tri4d/decode

 steps/train_sat.sh 1800 9000 data/train_vtln data/lang exp/tri4d exp/tri5d
 utils/mkgraph.sh data/lang exp/tri5d exp/tri5d/graph 
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5d/graph data/test_vtln exp/tri5d/decode 

 utils/mkgraph.sh data/lang_ug exp/tri5d exp/tri5d/graph_ug
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5d/graph_ug data/test_vtln exp/tri5d/decode_ug
)
