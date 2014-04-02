
all:

EXTRA_CXXFLAGS = -Wno-mismatched-tags -Wno-sign-compare
EXTRA_LDLIBS = -L../../tools/openfst/lib/fst/ -lfstfar -lfstfarscript
#-L../../tools/openfst-1.3.2/src/extensions/far/.libs \

include ../kaldi.mk

BINFILES = far-to-lattice  lattice-filter lattice-filter-text lattice-to-far

lattice-filter: exec-stream.o

lattice-filter-text: exec-stream.o

TESTFILES =

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../thread/kaldi-thread.a \
					../base/kaldi-base.a

include ../makefiles/default_rules.mk

