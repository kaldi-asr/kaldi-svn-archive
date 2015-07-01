
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare


OBJFILES = kaldi-durmod.o
LIBNAME = kaldi-durmod

ADDLIBS = ../lat/kaldi-lat.a ../tree/kaldi-tree.a \
					../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk

