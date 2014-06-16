
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = nnet2v-get-egs nnet2v-distort-egs

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet2/kaldi-nnet2.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a ../thread/kaldi-thread.a \
         ../cudamatrix/kaldi-cudamatrix.a ../feat/kaldi-feat.a ../matrix/kaldi-matrix.a \
         ../util/kaldi-util.a ../base/kaldi-base.a  

include ../makefiles/default_rules.mk
