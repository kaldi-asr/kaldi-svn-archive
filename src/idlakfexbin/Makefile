
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

ifeq ($(IDLAK),true)
	BINFILES = idlakfex
	OBJFILES =
else
	BINFILES =
	OBJFILES =
endif

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../idlakfex/idlakfex.a ../idlaktxp/idlaktxp.a ../util/kaldi-util.a ../base/kaldi-base.a 

# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

