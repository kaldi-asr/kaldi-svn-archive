#!/bin/bash

set -e
set -o pipefail
set -u

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh
parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.
train_stage=-100
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

./local/run_anydecode.sh bolt_dev &
./local/run_anydecode.sh bolt_tune &
./local/run_anydecode.sh bolt_test &
