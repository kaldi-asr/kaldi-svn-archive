all:

EXTRA_CXXFLAGS = -I$(PCREROOT)/include -I$(EXPATROOT)/include -I$(PUJIXMLROOT)/src
EXTRA_LDLIBS = $(PCREROOT)/lib/libpcre.a $(EXPATROOT)/lib/libexpat.a $(PUJIXMLROOT)/scripts/libpugixml.a
OPENFST_CXXFLAGS =
OPENFST_LDLIBS =

include ../kaldi.mk

TESTFILES = idlak-mod-test

OBJFILES = txpconfig.o txpxmldata.o txputf8.o txppcre.o txpnrules.o txppos.o \
	   txppbreak.o txpsylmax.o txplexicon.o txplts.o txpmodule.o txpfexspec.o \
	   fexfunctions.o fexfunctionscatalog.o mod-tokenise.o \
	   mod-postag.o mod-pauses.o mod-phrasing.o mod-pronounce.o mod-syllabify.o mod-fex.o

LIBNAME = idlak-txp
#LIBFILE = idlaktxp.a

ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

