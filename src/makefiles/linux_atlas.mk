# You have to make sure ATLASLIBS is set...

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

ifndef ATLASINC
$(error ATLASINC not defined.)
endif

ifndef ATLASLIBS
$(error ATLASLIBS not defined.)
endif

ifeq ($(IDLAK),true)
CXXFLAGS = -msse -msse2 -Wall -I.. \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I$(ATLASINC) \
      -I$(FSTROOT)/include \
      -I$(PCREROOT)/include \
      -I$(EXPATROOT)/include \
      -I$(PUJIXMLROOT)/src \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 
else
CXXFLAGS = -msse -msse2 -Wall -I.. \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I$(ATLASINC) \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 
endif

LDFLAGS = -rdynamic

ifeq ($(IDLAK),true)
	LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a $(PCREROOT)/lib/libpcrecpp.a $(PCREROOT)/lib/libpcre.a $(PUJIXMLROOT)/scripts/libpugixml.a $(EXPATROOT)/lib/libexpat.a -ldl $(ATLASLIBS) -lm -lpthread
else
	LDLIBS = $(EXTRA_LDLIBS) $(FSTROOT)/lib/libfst.a -ldl $(ATLASLIBS) -lm -lpthread
endif
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
