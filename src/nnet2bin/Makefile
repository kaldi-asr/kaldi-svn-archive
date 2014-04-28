
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet-randomize-frames nnet2-info nnet2-init \
   nnet2-train-simple nnet2-train-ensemble nnet2-train-transitions nnet2-latgen-faster nnet2-copy \
   nnet2-am-init nnet2-insert nnet2-align-compiled \
   nnet2-compute-prob nnet2-copy-egs nnet-am-shrink nnet-shrink nnet2-combine nnet-combine-a \
   nnet2-average nnet2-compute nnet2-mixup  \
   nnet2-train-lbfgs nnet2-get-egs nnet2-train-parallel nnet-gradient \
   nnet-get-preconditioner nnet-precondition nnet-select-egs nnet2-combine-fast \
   nnet2-shuffle-egs nnet2-fix nnet-logprob  nnet-logprob2 \
   nnet-logprob2-parallel nnet-logprob-parallel nnet-am-stats nnet-am-rescale \
   nnet-am-limit-rank nnet2-latgen-faster-parallel nnet2-to-raw-nnet2 \
   raw-nnet-concat nnet-am-limit-rank-final \
   nnet2-get-feature-transform nnet-compute-from-egs \
   nnet2-widen nnet2-show-progress nnet-limit-degradation \
   nnet2-get-feature-transform-multi nnet-copy-egs-discriminative \
   nnet2-get-egs-discriminative nnet-shuffle-egs-discriminative \
   nnet-compare-hash-discriminative nnet2-combine-egs-discriminative \
   nnet2-train-discriminative-simple nnet2-train-discriminative-parallel \
   nnet2-modify-learning-rates nnet2-subset-egs nnet2-normalize-stddev nnet-perturb-egs \
   nnet-perturb-egs-fmllr nnet-get-weighted-egs nnet2-adjust-priors \
   cuda-compiled nnet2-replace-last-layers 

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet2/kaldi-nnet2.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a ../thread/kaldi-thread.a \
         ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
         ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
