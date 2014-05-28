

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =

include ../kaldi.mk


TESTFILES = matrix-lib-test kaldi-gpsr-test kaldi-tensor-test

OBJFILES = kaldi-matrix.o kaldi-vector.o packed-matrix.o sp-matrix.o tp-matrix.o \
           matrix-functions.o qr.o srfft.o kaldi-gpsr.o compressed-matrix.o \
           optimization.o kaldi-tensor.o kaldi-tensor-internals.o

LIBNAME = kaldi-matrix

ADDLIBS = ../base/kaldi-base.a

include ../makefiles/default_rules.mk

