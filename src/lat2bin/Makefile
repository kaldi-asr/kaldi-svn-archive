KALDI_ROOT=../../

all: lattice-filter lattice-info lattice4-arc.so compactlattice44-arc.so lattice-filter-text

ifndef KALDI_ROOT
$(error KALDI_ROOT is not set)
endif

include $(KALDI_ROOT)/src/kaldi.mk
CXXFLAGS+=-I${KALDI_ROOT}/src/ -I${OPENLAT_ROOT}/src/include/

lattice4-arc.so: lattice-arc.o
	$(CXX)  $^ -o $@  $(LDFLAGS) $(LDLIBS) -shared

compactlattice44-arc.so: compact-lattice-arc.o
	$(CXX)  $^ -o $@  $(LDFLAGS) $(LDLIBS) -shared


lattice-filter: lattice-filter.o exec-stream.o
	$(CXX)  $^ -o $@  $(LDFLAGS) $(LDLIBS) $(KALDI_ROOT)/src/lat/kaldi-lat.a \
		$(KALDI_ROOT)/src/base/kaldi-base.a 	$(KALDI_ROOT)/src/util/kaldi-util.a \

lattice-filter-text: lattice-filter-text.o exec-stream.o
	$(CXX)  $^ -o $@  $(LDFLAGS) $(LDLIBS) $(KALDI_ROOT)/src/lat/kaldi-lat.a \
		$(KALDI_ROOT)/src/base/kaldi-base.a 	$(KALDI_ROOT)/src/util/kaldi-util.a \


lattice-info: lattice-info.o
	$(CXX)  $^ -o $@  $(LDFLAGS) $(LDLIBS) $(KALDI_ROOT)/src/lat/kaldi-lat.a \
		$(KALDI_ROOT)/src/base/kaldi-base.a 	$(KALDI_ROOT)/src/util/kaldi-util.a

clean:
	rm -rf *.o *.so lattice-filter-text lattice-info

%.o:%.cc
	$(CXX) $(CXXFLAGS) -c $<

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $<
