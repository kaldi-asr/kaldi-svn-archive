

all:

EXTRA_CXXFLAGS = -I$(PCREROOT)/include -I$(EXPATROOT)/include -I$(PUJIXMLROOT)/src
EXTRA_LDLIBS = $(PCREROOT)/lib/libpcre.a $(EXPATROOT)/lib/libexpat.a $(PUJIXMLROOT)/scripts/libpugixml.a

include ../kaldi.mk

TESTFILES = idlak-mod-test

OBJFILES = txpconfig.o txpxmldata.o txputf8.o txppcre.o txpnrules.o txppos.o \
	   txppbreak.o txpsylmax.o txplexicon.o txplts.o txpmodule.o txpfexspec.o \
	   fexfunctions.o fexfunctionscatalog.o mod-tokenise.o \
	   mod-postag.o mod-pauses.o mod-phrasing.o mod-pronounce.o mod-syllabify.o mod-fex.o

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

test_compile: $(TESTFILES)

test: test_compile
	@result=0; for x in $(TESTFILES); do echo -n "Running $$x ..."; ./$$x >/dev/null 2>&1; if [ $$? -ne 0 ]; then echo "... FAIL"; result=1; else echo "... SUCCESS";  fi;  done; exit $$result


.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

