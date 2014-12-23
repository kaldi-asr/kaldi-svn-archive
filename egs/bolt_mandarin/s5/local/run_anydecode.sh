#!/bin/bash 
set -e
set -o pipefail
. ./cmd.sh
. ./path.sh
. ./conf.sh

# dataset_id should be like: bolt_dev
dataset_id=$1
data=data/$dataset_id
decode_nj=12
graph=exp/tri5a/graph

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")

# SAT decode
if [ -f exp/tri5a/.done ]; then
  decode=exp/tri5a/decode_${dataset_id}
  #By default, tri5a system finished decoding at an earlier stage.
  if false; then
    echo "Decoding SAT system"
    steps/decode_fmllr_extra.sh --nj $decode_nj --cmd "$decode_cmd" "${extra_decoding_opts[@]}"\
       --config conf/decode.config  --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
       $graph $data $decode || exit 1;
    touch $decode/.done
  fi
  local/score.sh --cmd "$train_cmd" $data $graph $decode
fi

# DNN decode
if [ -f exp/tri6_nnet/.done ]; then
  decode=exp/tri6_nnet/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    echo "Decoding DNN system"
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $decode_nj \
      --config conf/decode.config --transform-dir exp/tri5a/decode_${dataset_id} \
      $graph $data $decode
    touch $decode/.done
  fi
  # local/score.sh --cmd "$train_cmd" $data $graph $decode
fi

# DNN mpe decode
if [ -f exp/tri6_nnet_mpe/.done ]; then
  for epoch in 4 5 6; do
    decode=exp/tri6_nnet_mpe/decode_${dataset_id}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      echo "decoding DNN mpe systems"
      mkdir -p $decode
      steps/nnet2/decode.sh  \
        --cmd "$decode_cmd" --nj $decode_nj --iter epoch$epoch \
        "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5a/decode_${dataset_id} \
        $graph $data $decode | tee $decode/decode.log
      touch $decode/.done
    fi
      # rescore with large LM
    if [ ! -f ${decode}_large/.done ]; then
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test data/lang_test_large \
          $data $decode ${decode}_large
      touch ${decode}_large/.done
    fi
    # local/score.sh --cmd "$train_cmd" $data $graph ${decode}_large
  done 
fi

# DNN Multisplice decode
if [ -f exp/tri6_nnet_multisplice/.done ]; then
  decode=exp/tri6_nnet_multisplice/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    echo "decoding DNN multisplice system"
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $decode_nj \
      --config conf/decode.config --transform-dir exp/tri5a/decode_${dataset_id} \
      $graph $data $decode
    touch $decode/.done
  fi
  # local/score.sh --cmd "$train_cmd" $data $graph $decode
fi

# DNN Multisplice mpe decode
if [ -f exp/tri6_nnet_multisplice_small_mpe/.done ]; then
  for epoch in 1 2 3 4 5 6; do
    decode=exp/tri6_nnet_multisplice_small_mpe/decode_${dataset_id}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      echo "decoding DNN multisplice mpe system"
      mkdir -p $decode
      steps/nnet2/decode.sh  \
        --cmd "$decode_cmd" --nj $decode_nj --iter epoch$epoch \
        "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5a/decode_${dataset_id} \
        $graph $data $decode | tee $decode/decode.log
      touch $decode/.done
    fi
    # rescore with large LM
    if [ ! -f ${decode}_large/.done ]; then
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test data/lang_test_large \
        $data $decode ${decode}_large
      touch ${decode}_large/.done
    fi
    # local/score.sh --cmd "$train_cmd" $data $graph $decode
  done 
fi

#  comb=exp/${dataset_id}_comb
#  if [ ! -f $comb/.done ]; then
#   echo "combining systems"
#   # Sys. Combination: DNN and DNN Multisplice
#   decode_dir1=exp/tri6_nnet_mpe/decode_${dataset_id}_epoch4
#   decode_dir2=exp/tri6_nnet_multisplice_mpe/decode_${dataset_id}_epoch4
#   systems="${decode_dir1}:0 ${decode_dir2}:0"
#   
#   local/score_combine.sh --skip-scoring false \
#     --cmd "$decode_cmd" --word-ins-penalty 0.5 \
#     $data $graph $systems $comb
#   touch $comb/.done
#  fi
