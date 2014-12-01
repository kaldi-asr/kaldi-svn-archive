#!/bin/bash 
# Copyright 2014  Xiaohui Zhang
# Apache 2.0.

set -e
set -o pipefail
. ./cmd.sh
. ./path.sh
. ./conf.sh

dataset_id=PROGRESS
data=data/$dataset_id
decode_nj=64
graph=exp/tri5a/graph
decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
opt_mpe_epoch=4
opt_lmwt=15
SortingProgram=`which hubscr.pl` || SortingProgram=$KALDI_ROOT/tools/sctk/bin/hubscr.pl

./local/prepare_eval_data.sh
# SAT decode
if [ -f exp/tri5a/.done ]; then
  decode=exp/tri5a/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    echo "Decoding SAT system"
    steps/decode_fmllr_extra.sh --nj $decode_nj --cmd "$decode_cmd" "${extra_decoding_opts[@]}"\
       --skip-scoring true \
       --config conf/decode.config  --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
       $graph $data $decode || exit 1;
    local/lattice_to_ctm.sh --min_lmwt 8 --max_lmwt 18 --cmd "$train_cmd" $data $graph $decode
    touch $decode/.done
  fi
  # post-processing of the optimal ctm
  opt_ctm=${decode}/score_${opt_lmwt}/${dataset_id}.ctm
  $SortingProgram sortCTM <$opt_ctm  >${opt_ctm}.sorted 
  cp ${opt_ctm}.sorted ${dataset_id}.sat.ctm 
fi

# DNN mpe decode
if [ -f exp/tri6_nnet_mpe/.done ]; then
  for epoch in $opt_mpe_epoch; do
    decode=exp/tri6_nnet_mpe/decode_${dataset_id}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      echo "decoding DNN mpe systems"
      mkdir -p $decode
      steps/nnet2/decode.sh  \
        --skip-scoring true \
        --cmd "$decode_cmd" --nj $decode_nj --iter epoch$epoch \
        "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5a/decode_${dataset_id} \
        $graph $data $decode | tee $decode/decode.log
      local/lattice_to_ctm.sh --min_lmwt 8 --max_lmwt 18 --cmd "$train_cmd" $data $graph $decode
      touch $decode/.done
    fi
      # rescore with large LM
    if [ ! -f ${decode}_large/.done ]; then
      steps/lmrescore_const_arpa.sh \
          --skip-scoring true \
          --cmd "$decode_cmd" data/lang_test data/lang_test_large \
          $data $decode ${decode}_large
      touch ${decode}_large/.done
    fi   
    local/lattice_to_ctm.sh --min_lmwt 14 --max_lmwt 18 --cmd "$train_cmd" $data $graph ${decode}_large
    # post-processing of the optimal ctm
    opt_ctm=${decode}_large/score_${opt_lmwt}/${dataset_id}.ctm
    $SortingProgram sortCTM <$opt_ctm  >${opt_ctm}.sorted 
    cp ${opt_ctm}.sorted ${dataset_id}.dnn_mpe.ctm 
  done     
fi        
          

# comb=exp/${dataset_id}_comb
# if [ ! -f $comb/.done ]; then
#  echo "combining systems"
#  # Sys. Combination: DNN and DNN Multisplice
#  decode_dir1=exp/tri6_nnet/decode_${dataset_id}
#  decode_dir2=exp/tri6_nnet_multisplice/decode_${dataset_id}
#  systems="${decode_dir1}:0 ${decode_dir2}:0"
#  
#  local/score_combine.sh --skip-scoring false \
#    --cmd "$decode_cmd" --word-ins-penalty 0.5 \
#    $data $graph $systems $comb
#  touch $comb/.done
# fi
