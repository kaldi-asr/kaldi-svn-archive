#!/bin/bash

. cmd.sh


# This example runs on top of LDA+MLLT+SAT features using p-norm nonlinearity.
# (after reducing splice-width from 7 to 5)

# prepare combined posteriors for multitask learning
steps/prepare_posts.sh data/train exp/tri3b_ali post

(  steps/nnet2/train_pnorm_multitask.sh --splice-width 5 \
     --num-jobs-nnet 4 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --num-hidden-layers 2 \
     --num-epochs 20 --num-epochs-extra 10 \
     --add-layers-period 1 \
     --mix-up 0 \
     --cmd "$decode_cmd" \
     --pnorm-input-dim 1000 \
     --pnorm-output-dim 200 \
     --stage 27 \
     --use-alignment false \
     data/train data/lang exp/tri3b_ali post/combinedpost exp/nnet_multask  || exit 1

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type lda \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test exp/nnet_multask/decode

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type lda \
     --transform-dir exp/tri3b/decode_ug \
     exp/tri3b/graph_ug data/test exp/nnet_multask/decode_ug
 ) 

