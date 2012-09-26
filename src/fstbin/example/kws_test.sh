#!/bin/bash

if [ ${PWD##*/} != "example" ];then
  echo "This script should be run from example directory."
  exit 1;
fi

../../latbin/lattice-to-fst --acoustic-scale=1.0 --lm-scale=1.0 ark,t:utts.txt ark,t:- | ../fstpush-kaldi --push-in-log=true ark,t:- ark,t:- | ../fsttoindex isyms.txt ark,t:- ark,t:- | ../fstdeterminizestar --use-log=true ark,t:- ark,t:- |  ../fstpush-kaldi --remove-total-weight=false --push-in-log=true ark,t:- ark,t:- | ../fstminimize-kaldi --encode ark,t:- ark,t:- | ../fstunion-kaldi ark,t:- | ../../../tools/openfst/bin/fstrmepsilon | ../fstdeterminize-kws > test.fst

../../../tools/openfst/bin/fstequal test.fst index.fst
OUT=$?

if  [ $OUT -eq 0 ]; then
  echo "Test has succeeded."
else
  echo "Error: the newly created index is different from the reference fst."
fi

