#!/bin/bash

set -e
set -o pipefail
set -u
set -x

. ./path.sh
. ./cmd.sh

parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.
train_stage=-100
multisplice=false
if $multisplice; then
  nnet_dir=tri6_nnet_multisplice
else
  nnet_dir=tri6_nnet
fi
. utils/parse_options.sh
# Wait for cross-entropy training.
echo "Waiting till exp/${nnet_dir}/.done exists...."
while [ ! -f exp/${nnet_dir}/.done ]; do sleep 30; done
echo "...done waiting for exp/${nnet_dir}/.done"

# Generate denominator lattices.
if [ ! -f exp/${nnet_dir}_denlats/.done ]; then
  steps/nnet2/make_denlats.sh --num-threads 1 \
    --nj 30 --sub-split 30 \
    --transform-dir exp/tri5a_ali \
    data/train data/lang exp/${nnet_dir} exp/${nnet_dir}_denlats || exit 1
  touch exp/${nnet_dir}_denlats/.done
fi

# Generate alignment.
if [ ! -f exp/${nnet_dir}_ali/.done ]; then
  steps/nnet2/align.sh --use-gpu yes \
    --cmd "$decode_cmd $parallel_opts" \
    --transform-dir exp/tri5a_ali --nj 30 \
    data/train data/lang exp/${nnet_dir} exp/${nnet_dir}_ali || exit 1
  touch exp/${nnet_dir}_ali/.done
fi

if [ ! -f exp/${nnet_dir}_mpe/.done ]; then
  steps/nnet2/train_discriminative.sh \
    --stage $train_stage --cmd "$decode_cmd" \
    --parallel-opts "$parallel_opts" \
    --learning-rate 0.00009 \
    --modify-learning-rates true \
    --num-epochs 6 --cleanup false \
    --transform-dir exp/tri5a_ali \
    --num-jobs-nnet 8 --num-threads 1 \
    --stage $train_stage \
    data/train data/lang \
    exp/${nnet_dir}_ali exp/${nnet_dir}_denlats exp/${nnet_dir}/final.mdl exp/${nnet_dir}_mpe || exit 1
  touch exp/${nnet_dir}_mpe/.done
fi

./local/run_anydecode.sh bolt_dev &
./local/run_anydecode.sh bolt_tune &
