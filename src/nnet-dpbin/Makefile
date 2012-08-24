
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = nnet1-init nnet1-train nnet1-add-layer nnet1-info nnet1-mixup

OBJFILES =

all:  $(BINFILES)

TESTFILES =

$(BINFILES): ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../nnet-dp/kaldi-nnet-dp.a \
    ../gmm/kaldi-gmm.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
    ../base/kaldi-base.a ../thread/kaldi-thread.a



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || { echo "***test $$x failed***"; exit 1; }; done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

