#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# Begin configuration.
cmd=run.pl
config=
stage=-4
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
mllt_iters="2 4 6 12";
num_iters=35    # Number of iterations of training
max_iter_inc=25  # Last iter to increase #Gauss on.
dim=40
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.2 # Exponent for number of gaussians according to occurrence counts
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
splice_opts=
# End configuration.

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: local/train_lda_mllt_notree.sh [options] <#gauss> <data> <lang> <alignments> <dir>"
  echo " e.g.: local/train_lda_mllt_notree.sh 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-jobs>                                  # how many parallel jobs to run"
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

totgauss=$1
data=$2
lang=$3
alidir=$4
dir=$5

for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_lda_mllt.sh: no such file $f" && exit 1;
done

numgauss=`gmm-info $alidir/final.mdl | grep -w pdfs | awk '{print $NF}'`
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter #gauss increment
oov=`cat $lang/oov.int` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;

mkdir -p $dir/log
echo $nj >$dir/num_jobs
echo "$splice_opts" >$dir/splice_opts # keep track of frame-splicing options
           # so that later stages of system building can know what they were.

sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;


splicedfeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
# Note: $feats gets overwritten later in the script.
feats="$splicedfeats transform-feats $dir/0.mat ark:- ark:- |"



if [ $stage -le -4 ]; then
  echo "Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
  est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;
  rm $dir/lda.*.acc
fi

cur_lda_iter=0

cp $alidir/tree $dir


if [ $stage -le -3 ]; then
  echo "Initializing the model"
  $cmd JOB=1 $dir/log/init_model.log \
    gmm-init-model-flat $dir/tree $lang/topo $dir/1.mdl \
      "$feats subset-feats ark:- ark:- |" || exit 1;
fi    

if [ $stage -le -1 ]; then
  # Copy the alignments.
  cp $alidir/ali.*.gz $dir/
fi

if [ $stage -le 0 ]; then
  echo "Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi


x=1
while [ $x -lt $num_iters ]; do
  echo Training pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi
  if echo $mllt_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo "Estimating MLLT"
      $cmd JOB=1:$nj $dir/log/macc.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-acc-mllt --rand-prune=$randprune  $dir/$x.mdl "$feats" ark:- $dir/$x.JOB.macc \
        || exit 1;
      est-mllt $dir/$x.mat.new $dir/$x.*.macc 2> $dir/log/mupdate.$x.log || exit 1;
      gmm-transform-means  $dir/$x.mat.new $dir/$x.mdl $dir/$x.mdl \
        2> $dir/log/transform_means.$x.log || exit 1;
      compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_lda_iter.mat $dir/$x.mat || exit 1;
      rm $dir/$x.*.macc
    fi
    feats="$splicedfeats transform-feats $dir/$x.mat ark:- ark:- |"
    cur_lda_iter=$x
  fi

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power \
        $dir/$x.mdl "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done

rm $dir/final.{mdl,mat,occs} 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs
ln -s $cur_lda_iter.mat $dir/final.mat

# Summarize warning messages...

utils/summarize_warnings.pl $dir/log

echo Done training system with LDA+MLLT features in $dir
