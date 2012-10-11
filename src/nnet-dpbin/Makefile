
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = nnet1-init nnet1-train nnet1-add-layer nnet1-info nnet1-mixup \
    nnet1-copy nnet1-latgen-faster nnet1-avg

OBJFILES =

all:  $(BINFILES)

TESTFILES =

$(BINFILES): ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a ../hmm/kaldi-hmm.a \
     ../tree/kaldi-tree.a ../nnet-dp/kaldi-nnet-dp.a ../gmm/kaldi-gmm.a \
    ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
    ../base/kaldi-base.a



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o $(TESTFILES) $(BINFILES)

test_compile: $(TESTFILES)
  
test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

