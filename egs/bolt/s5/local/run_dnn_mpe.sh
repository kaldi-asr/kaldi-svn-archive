#!/bin/bash

set -e
set -o pipefail
set -u

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh
parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

# Wait for cross-entropy training.
echo "Waiting till exp/tri6_nnet/.done exists...."
while [ ! -f exp/tri6_nnet/.done ]; do sleep 30; done
echo "...done waiting for exp/tri6_nnet/.done"

# Generate denominator lattices.
if [ ! -f exp/tri6_nnet_denlats/.done ]; then
  steps/nnet2/make_denlats.sh --num-threads 1 \
    --nj 30 --sub-split 30 \
    --transform-dir exp/tri5a_ali \
    data/train data/lang exp/tri6_nnet exp/tri6_nnet_denlats || exit 1
 
  touch exp/tri6_nnet_denlats/.done
fi

# Generate alignment.
if [ ! -f exp/tri6_nnet_ali/.done ]; then
  steps/nnet2/align.sh --use-gpu yes \
    --cmd "$decode_cmd $parallel_opts" \
    --transform-dir exp/tri5a_ali --nj 30 \
    data/train data/lang exp/tri6_nnet exp/tri6_nnet_ali || exit 1
  touch exp/tri6_nnet_ali/.done
fi

train_stage=-100
if [ ! -f exp/tri6_nnet_mpe/.done ]; then
  steps/nnet2/train_discriminative.sh \
    --stage $train_stage --cmd "$decode_cmd" \
    --stage -10 \
    --parallel-opts "$parallel_opts" \
    --learning-rate 0.00009 \
    --modify-learning-rates true \
    --num-epochs 4 --cleanup false \
    --transform-dir exp/tri5a_ali \
    --num-jobs-nnet 8 --num-threads 1 \
    data/train data/lang \
    exp/tri6_nnet_ali exp/tri6_nnet_denlats exp/tri6_nnet/final.mdl exp/tri6_nnet_mpe || exit 1
  touch exp/tri6_nnet_mpe/.done
fi

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")
for epoch in 1 2 3 4; do
  decode=exp/tri6_nnet_mpe_new/decode_dev_epoch$epoch
  if [ ! -f $decode/.done ]; then
    mkdir -p $decode
    steps/nnet2/decode.sh  \
      --cmd "$decode_cmd" --nj 30 --iter epoch$epoch \
      "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5a/decode_dev \
      exp/tri5a/graph data/dev $decode | tee $decode/decode.log
    touch $decode/.done
  fi
done
