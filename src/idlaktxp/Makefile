

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =

include ../kaldi.mk

TESTFILES = idlak-mod-test

OBJFILES = txpconfig.o txpxmldata.o txputf8.o txppcre.o txpnrules.o txppos.o txppbreak.o txpsylmax.o txplexicon.o txplts.o txpmodule.o mod-tokenise.o mod-postag.o mod-pauses.o mod-phrasing.o mod-pronounce.o mod-syllabify.o

LIBFILE = idlaktxp.a

all:  $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBFILE) $(OBJFILES)
	$(RANLIB) $(LIBFILE)


$(TESTFILES): $(LIBFILE) ../util/kaldi-util.a ../base/kaldi-base.a


# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	-rm -f *.o *.a $(TESTFILES) tmp* *.tmp

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || (echo "***test $$x failed***"; exit 1); done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

