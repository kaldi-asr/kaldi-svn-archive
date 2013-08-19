
all:

include ../kaldi.mk

# note, build-tree-utils-test also tests build-tree-questions.cc

TESTFILES = event-map-test context-dep-test build-tree-utils-test \
						cluster-utils-test build-tree-test topo-tree-test


OBJFILES = event-map.o context-dep.o clusterable-classes.o cluster-utils.o \
					 build-tree-utils.o build-tree.o build-tree-questions.o tree-renderer.o \
           topo-tree.o build-topo-tree.o

LIBNAME = kaldi-tree
ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
