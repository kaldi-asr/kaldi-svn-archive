
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = gen-durmod-egs
OBJFILES =



TESTFILES =


ADDLIBS = ../durmod/kaldi-durmod.a ../nnet2/kaldi-nnet2.a \
					../lat/kaldi-lat.a ../fstext/kaldi-fstext.a \
					../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
					../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
