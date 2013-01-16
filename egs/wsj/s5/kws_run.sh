#!/bin/bash

. path.sh
. mirko_cmd.sh

if [ $# -lt 9 ] ; then
  echo "Bad number of arguments"
  exit 1
fi

dataset=${1}
lang=${2}
latticepath=${3}
indexing=${4}
acfactor=${5}
lmfactor=${6}
globalfactor=${7}
clustering=${8}
resultdir=${9}

# Data settings
datadir=data/${dataset}
evaldir=data/eval/${dataset}

# Input settings
basedir=`dirname $latticepath`
latticedirname=`basename $latticepath`
decodedir=$basedir/${latticedirname}

# Computing "real" factors
acwt=`echo "scale=8;1/$acfactor" | bc`
lmwt=`echo "scale=8;1/$lmfactor" | bc`
acwt=`echo "scale=8;$acwt * $globalfactor" | bc`
lmwt=`echo "scale=8;$lmwt * $globalfactor" | bc`

# KWS output directory
expname=AC${acfactor}_LM${lmfactor}_GL${globalfactor}_CL${clustering}
dir=exp/tri3b_but69/${latticedirname}/kws/${expname}

mkdir -p $dir

if $indexing ; then
  # Indexing lattices.
  steps/make_index.sh --cmd "$decode_cmd" --acwt $acwt --lmwt $lmwt --clustering $clustering \
    $datadir/kws/ \
    data/lang/ \
    $decodedir/ \
    $dir

fi

# Searching keywords.
steps/search_index.sh --cmd "$decode_cmd" --char true \
  $datadir/kws \
  $dir/

# Gathering keyword spotting results in one file.
cat  $dir/result.* > $dir/kws_results.txt

mkdir -p $dir/logadd

# Postprocessing.
local/kws_postprocessing.py --postproc logadd $dir/kws_results.txt > $dir/logadd/kws_results_final.txt

