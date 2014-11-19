#!/bin/bash 
set -e
set -o pipefail
. ./cmd.sh
. ./path.sh
. ./conf.sh

# dataset_id should be like: dolt_dev
dataset_id=$1
data=data/$dataset_id
decode_nj=30
graph=exp/tri5a/graph

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")

# SAT decode
if [ -f exp/tri5a/.done ]; then
  decode=exp/tri5a/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    steps/decode_fmllr_extra.sh --nj $decode_nj --cmd "$decode_cmd" "${extra_decoding_opts[@]}"\
       --config conf/decode.config  --scoring_opts "--min_lmwt 8 --max_lmwt 14 "\
       $graph $data $decode || exit 1;
  fi
fi

# DNN decode
if [ -f exp/tri6_nnet/.done ]; then
  decode=exp/tri6_nnet/decode_${dataset_id}
  if [ ! -f $decode/.done ]; then
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $decode_nj \
      --config conf/decode.config --transform-dir exp/tri5a/decode_${dataset_id} \
      $graph $data $decode
    # local/score.sh --cmd "$train_cmd" $data $graph $decode
  fi
fi

# DNN mpe decode
if [ -f exp/tri6_nnet_mpe/.done ]; then
  for epoch in 1 2 3 4; do
    decode=exp/tri6_nnet_mpe/decode_${dataset_id}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      mkdir -p $decode
      steps/nnet2/decode.sh  \
        --cmd "$decode_cmd" --nj $decode_nj --iter epoch$epoch \
        "${decode_extra_opts[@]}" \
        --transform-dir exp/tri5a/decode_${dataset_id} \
        $graph $data $decode | tee $decode/decode.log
      # rescore with large LM
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test data/lang_test_large \
          $data exp/tri6_nnet_mpe/decode_${dataset_id}_epoch$iter \
                exp/tri6_nnet_mpe/decode_${dataset_id}_epoch${iter}_large
      touch $decode/.done
      # local/score.sh --cmd "$train_cmd" $data $graph $decode
    fi
  done 
fi
