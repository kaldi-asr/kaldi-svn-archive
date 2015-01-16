#!/bin/bash

# Copyright 2014 Xiaohui Zhang
# Apache 2.0.
# This srcipt computes posteriors of a SGMM system. First we align the data using the existing SGMM model, 
# Then we compute log-likelihoods and priors of the data given the model, and then we sum them up to get
# the posteriors. All outputs (alignments, priors, log-likelihoods, posteriors) will be in <output-dir>. 

# Begin configuration section.
cmd=run.pl
stage=0
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
transform_dir=  # directory to find fMLLR transforms in.
cleanup=true

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
if [ -f cmd.sh ]; then . ./cmd.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/post/get_sgmm_post.sh <data-dir> <lang-dir> <src-dir> <output-dir>"
  echo "e.g.:  steps/post/get_sgmm_post.sh --transform-dir exp/tri3b data/train data/lang exp/tri3b exp/tri3b_post."
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4
mkdir -p $dir

# Check some files.
for f in  $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $srcdir/num_jobs`
cp $srcdir/num_jobs $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
sdata=$data/split$nj
gselect_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|"
num_leaves=`tree-info $srcdir/tree 2>/dev/null | awk '/num-pdfs/{print $NF}'`

## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
  echo "  but you are not providing the --transform-dir option during alignment."
fi
##


if [ $stage -le 0 ]; then
  echo "$0: Calling steps/align_sgmm.sh in order to get alignments and speaker vectors."
  steps/align_sgmm2.sh --nj $nj --cmd "$train_cmd" --transform-dir $transform_dir \
    --use-graphs false --use-gselect false $data $lang $srcdir $dir
fi

echo $num_leaves
if [ $stage -le 1 ]; then
  echo "$0: computing sgmm2 log likelihoods."
  $cmd JOB=1:$nj $dir/log/compute_loglike.JOB.log \
    sgmm2-compute $scale_opts "$gselect_opt"  --num_pdfs=$num_leaves --utt2spk=ark:$sdata/JOB/utt2spk --spk-vecs=ark:$dir/vecs.JOB \
      $srcdir/final.mdl "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,scp:$dir/loglike.JOB.ark,$dir/loglike.JOB.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing priors from alignments."
  ali-to-pdf $srcdir/final.mdl "ark:gunzip -c $dir/ali.*.gz|" ark:- | pdf-to-prior --num_pdfs=$num_leaves ark:- $dir/prior.vec || exit 1;
fi 

if [ $stage -le 3 ]; then
  echo "$0: computing sgmm2 posteriors."
  $cmd JOB=1:$nj $dir/log/compute_post.JOB.log \
    add-vec-to-rows $dir/prior.vec scp:$dir/loglike.JOB.scp ark:- \| \
      logprob-to-post ark:- ark,scp:$dir/post.JOB.ark,$dir/post.JOB.scp || exit 1;
fi

if [ $cleanup ]; then
  rm -rf $dir/loglike* 
fi

echo "$0: Finished computing SGMM posteriors."
