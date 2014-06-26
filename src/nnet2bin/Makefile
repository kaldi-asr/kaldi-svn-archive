
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet2-randomize-frames nnet2-info nnet2-init \
   nnet2-train-simple nnet2-train-ensemble nnet2-train-transitions nnet2-latgen-faster nnet2-copy \
   nnet2-am-init nnet2-insert nnet2-align-compiled \
   nnet2-compute-prob nnet2-copy-egs nnet2-am-shrink nnet2-shrink nnet2-combine nnet2-combine-a \
   nnet2-average nnet2-compute nnet2-mixup  \
   nnet2-train-lbfgs nnet2-get-egs nnet2-train-parallel nnet2-gradient \
   nnet2-get-preconditioner nnet2-precondition nnet2-select-egs nnet2-combine-fast \
   nnet2-shuffle-egs nnet2-fix nnet2-logprob  nnet2-logprob2 \
   nnet2-logprob2-parallel nnet2-logprob-parallel nnet2-am-stats nnet2-am-rescale \
   nnet2-am-limit-rank nnet2-latgen-faster-parallel nnet2-to-raw-nnet2 \
   raw-nnet2-concat raw-nnet2-info nnet2-am-limit-rank-final \
   nnet2-get-feature-transform nnet2-compute-from-egs \
   nnet2-widen nnet2-show-progress nnet2-limit-degradation \
   nnet2-get-feature-transform-multi nnet2-copy-egs-discriminative \
   nnet2-get-egs-discriminative nnet2-shuffle-egs-discriminative \
   nnet2-compare-hash-discriminative nnet2-combine-egs-discriminative \
   nnet2-train-discriminative-simple nnet2-train-discriminative-parallel \
   nnet2-modify-learning-rates nnet2-subset-egs nnet2-normalize-stddev nnet2-perturb-egs \
   nnet2-perturb-egs-fmllr nnet2-get-weighted-egs nnet2-adjust-priors \
   cuda-compiled nnet2-replace-last-layers 

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet2/kaldi-nnet2.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a ../thread/kaldi-thread.a \
         ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a \
         ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
