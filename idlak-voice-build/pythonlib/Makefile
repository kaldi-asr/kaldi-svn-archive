all:

include ../../src/kaldi.mk

TARGET     = idlakapi

SWIG       = swig
SWIGOPT    = -classic
CXX        = g++
CXXSRCS       = idlakapi.cc
INCLUDE    = -I../../src
LIBS       = ../../src/tree/kaldi-tree.a ../../src/matrix/kaldi-matrix.a ../../src/util/kaldi-util.a ../../src/base/kaldi-base.a 
INTERFACE  = idlakapi.i
CXXSHARED  = g++ -shared
SO         = .so

IWRAP      = $(INTERFACE:.i=_wrap.i)
ISRCS      = $(IWRAP:.i=.c)
ICXXSRCS   = $(IWRAP:.i=.cxx)
IOBJS      = $(IWRAP:.i=.o)

OBJS      = $(CXXSRCS:.cc=.o)

##################################################################
#####                       PYTHON                          ######
##################################################################

# Make sure these locate your Python installation
PYTHON_INCLUDE= -I/usr/include/python2.6
PYTHON_LIB    = 

# Extra Python specific dynamic linking options
PYTHON_DLNK   = -lpython2.6

# -----------------------------------------------------------------
# Build a C++ dynamically loadable module
# -----------------------------------------------------------------

all: $(SRCS)
	$(SWIG) -c++ -python $(SWIGOPT) $(INTERFACE)
	$(CXX) $(INCLUDE) $(PYTHON_INCLUDE) $(CXXFLAGS) -c $(CXXSRCS) $(ICXXSRCS)
	$(CXXSHARED) $(OBJS) $(IOBJS) $(PYTHON_DLNK) $(LIBS) $(LDLIBS) -o _$(TARGET)$(SO)
