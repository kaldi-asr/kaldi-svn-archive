#!/bin/bash

set -e
set -o pipefail

. ./path.sh
. ./cmd.sh

train_nj=64
dnn_parallel_opts="-l gpu=1"
dnn_denlats_extra_opts=(--num-threads 4 
                        --parallel-opts "-pe smp 4" 
                        --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=0.8G"
                      )

dnn_mpe_learning_rate=0.00008
dnn_mpe_last_layer_factor=0.1
dnn_mpe_retroactive=true
dnn_gpu_mpe_parallel_opts=(
                        --num-jobs-nnet 8 
                        --num-threads 1 \
                        --parallel-opts "-l gpu=1" 
                        --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G"
                        )
set -u
if false; then
# Generate denominator lattices.
steps/nnet2/make_denlats.sh "${dnn_denlats_extra_opts[@]}" \
  --nj $train_nj --sub-split $train_nj \
  --transform-dir exp/tri5b_ali \
  data/train data/lang exp/tri6b_dnn exp/tri6b_dnn_denlats || exit 1
 

# Generate alignment.
steps/nnet2/align.sh --use-gpu no  --cmd "$decode_cmd" \
  --transform-dir exp/tri5b_ali --nj $train_nj \
  data/train data/lang exp/tri6b_dnn exp/tri6b_dnn_ali || exit 1


steps/nnet2/train_discriminative.sh \
  --cmd "$decode_cmd" \
  --learning-rate $dnn_mpe_learning_rate \
  --modify-learning-rates true \
  --last-layer-factor $dnn_mpe_last_layer_factor \
  --num-epochs 4 --cleanup true \
  --retroactive true \
  --transform-dir exp/tri5b_ali \
  "${dnn_gpu_mpe_parallel_opts[@]}" data/train data/lang \
  exp/tri6b_dnn_ali exp/tri6b_dnn_denlats exp/tri6b_dnn/final.mdl exp/tri6b_dnn_mpe || exit 1


for epoch in 1 2 3 4; do
  decode=exp/tri6b_dnn_mpe/decode_tune_epoch$epoch
  mkdir -p $decode
  steps/nnet2/decode.sh --minimize true \
    --cmd "$decode_cmd" --nj 13 --iter epoch$epoch \
    --beam 16 --lattice-beam 8.5 \
    --transform-dir exp/tri5b/decode_tune \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri5b/graph data/tune $decode | tee $decode/decode.log
done

fi

steps/nnet2/train_discriminative.sh \
  --cmd "$decode_cmd" \
  --num-epochs 4 --cleanup true \
  --transform-dir exp/tri5b_ali \
  "${dnn_gpu_mpe_parallel_opts[@]}" data/train data/lang \
  exp/tri6b_dnn_ali exp/tri6b_dnn_denlats exp/tri6b_dnn/final.mdl exp/tri6b_dnn_mpe_2 || exit 1


for epoch in 1 2 3 4; do
  decode=exp/tri6b_dnn_mpe_2/decode_tune_epoch$epoch
  mkdir -p $decode
  steps/nnet2/decode.sh --minimize true \
    --cmd "$decode_cmd" --nj 13 --iter epoch$epoch \
    --beam 16 --lattice-beam 8.5 \
    --transform-dir exp/tri5b/decode_tune \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri5b/graph data/tune $decode | tee $decode/decode.log
done

