#!/bin/bash
# Copyright 2014 Mobvoi Inc. (Author: Xiaohui Zhang)
# Apache 2.0

# Get posteriors from a dnn.

# Begin configuration section.  
cmd="queue.pl -l arch=*64"
parallel_opts="-l gpu=1" # This is suitable for the CLSP network, you'll likely have to change it.
# Begin configuration.
transform_dir=
nj=50
raw=false
min_post=0.01
use_gpu="no" # yes|no|optionaly
# End configuration options.

[ $# -gt 0 ] && echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
[ -f cmd.sh ] && . ./cmd.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: $0 <data-dir> <src-dir> <post-dir>"
   echo "e.g.:  $0 data/train exp/tri1 exp/tri1_soft_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
nnetdir=$2
dir=$3
echo $data
echo $nnetdir
mkdir -p $dir/log
[ ! -z $transform_dir ] && nj=`cat $transform_dir/num_jobs` && echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;

cp $nnetdir/{tree,final.mdl} $dir || exit 1;



# Assume that final.mat and final.nnet are at nnetdir
nnet_lda=$nnetdir/final.mat
nnet=$nnetdir/final.mdl
for file in $nnet_lda $nnet; do
  if [ ! -f $file ] ; then
    echo "No such file $file";
    exit 1;
  fi
done

name=`basename $data`
sdata=$data/split$nj

mkdir -p $dir/log
echo $nj > $nnetdir/num_jobs
nnet_plice_opts=`cat $nnetdir/nnet_splice_opts 2>/dev/null`
splice_opts=`cat $nnetdir/splice_opts 2>/dev/null`
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Set up input features of nnet
if [ -f $nnetdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"
if [ $raw == "true" ]; then feat_type=raw; fi

case $feat_type in
  raw)  feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $nnetdir/final.mat ark:- ark:- |"
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ -f $transform_dir/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
  nj=`cat $transform_dir/num_jobs`
  echo $nj > $nnetdir/num_jobs
  cp $transform_dir/trans* $dir || exit 1;
fi  

if [ -f $transform_dir/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw-fMLLR transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/raw_trans.JOB ark:- ark:- |"
  cp $transform_dir/raw_trans* $dir || exit 1;
fi

echo "Making  scp and ark."
  $cmd $parallel_opts JOB=1:$nj $dir/log/get_post.JOB.log \
    nnet-am-compute --use-gpu=$use_gpu $nnet "$feats"  ark:- \| \
    prob-to-post --min-post=$min_post ark:- ark,scp:$dir/post.JOB.ark,$dir/post.JOB.scp || exit 1;
exit

### PREPARE FEATURE EXTRACTION PIPELINE
### Create the feature stream:
##feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
### Optionally add cmvn
##if [ -f $srcdir/norm_vars ]; then
##  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
##  [ ! -f $sdata/1/cmvn.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn.scp" && exit 1
##  feats="$feats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
##fi
### Optionally add deltas
##if [ -f $srcdir/delta_order ]; then
##  delta_order=$(cat $srcdir/delta_order)
##  feats="$feats add-deltas --delta-order=$delta_order ark:- ark:- |"
##fi
### Finally add feature_transform and the MLP
##feats="$feats nnet-forward --feature-transform=$feature_transform --no-softmax=false --use-gpu=$use_gpu $nnet ark:- ark:- |"
### feats="$feats nnet-forward --feature-transform=$feature_transform --no-softmax=false --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet ark:- ark:- |"
##
##
##echo "$0: generating posteriors for '$data' using nnet/model '$srcdir', putting alignments in '$dir'"
##if [ $stage -le 0 ]; then
##  $cmd JOB=1:$nj $dir/log/genpost.JOB.log \
##    prob-to-post "$feats" "ark:|gzip -c >$dir/post.JOB.gz" || exit 1;
##fi
##
