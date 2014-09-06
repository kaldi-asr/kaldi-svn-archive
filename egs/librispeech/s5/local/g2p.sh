#!/bin/bash

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# The only reason for the existence of this script is the inability of Sequitur
# to save the results of its work into a file, rather that print them on stdout

. path.sh || exit 1

[ -z "$PYTHON" ] && PYTHON=python2.7 

if [ $# -ne 3 ]; then
  echo "Usage: $0 <vocab> <g2p-model-dir> <out-lexicon>"
  echo "e.g.: $0 data/local/dict/g2p/vocab_autogen.1 /export/a15/vpanayotov/data/g2p data/local/dict/g2p/lexicon_autogen.1"
  echo ", where:"
  echo "    <vocab> - input vocabulary, that's words for which we want to generate pronunciations"
  echo "    <g2p-model-dir> - source directory where g2p model is located"
  echo "    <out-lexicon> - the output, i.e. the generated pronunciations"
  exit 1
fi

vocab=$1
g2p_model_dir=$2
out_lexicon=$3

[ ! -f $vocab ] && echo "Can't find the G2P input file: $vocab" && exit 1;

# Sequitur G2P executable
sequitur=$KALDI_ROOT/tools/sequitur/g2p.py
sequitur_path="$(dirname $sequitur)/lib/$PYTHON/site-packages"
sequitur_model=$g2p_model_dir/model-full.5

[ ! -f  $sequitur ] && \
  echo "Can't find the Sequitur G2P script. Please check $KALDI_ROOT/tools for installation script and instructions" && \
  exit 1;

[ ! -d $sequitur_path ] && echo "Can't find '$sequitur_path' - please fix your Sequitur installation" && exit 1
[ ! -f $sequitur_model ] && echo "Can't find the Sequitur model file: $sequitur_model" && exit 1

PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
  --model=$sequitur_model --apply $vocab \
  >$out_lexicon || exit 1

exit 0
