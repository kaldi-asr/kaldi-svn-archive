#!/bin/bash

mkdir -p orig_includes
(  # This downloadds clapack.h from netlib and cblas.h from ATLAS.
  cd orig_includes
  wget http://www.netlib.org/clapack/clapack.h
  wget http://sourceforge.net/projects/math-atlas/files/Stable/3.8.3/atlas3.8.3.tar.gz
  tar xozf atlas3.8.3.tar.gz ATLAS/include/cblas.h || exit 1;
  mv ATLAS/include/cblas.h . || exit 1;
  rm -r ATLAS/include
  # rm atlas3.8.3.tar.gz
)

mkdir -p const_headers

./add_const_guided.pl orig_includes/cblas.h  cblas.h > const_headers/cblas.h

## Actually it turns out I wasted my time in getting the scripts to work for
## CLAPACK: there is no const in the clapack.h from netlib anyway.
##./add_const_guided.pl orig_includes/clapack.h common_interface.h > const_headers/common_interface.h

