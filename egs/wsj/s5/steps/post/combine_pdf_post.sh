#!/bin/bash

# Copyright 2015 Xiaohui Zhang
# Apache 2.0.
# This srcipt computes posteriors of a SGMM system. First we align the data using the existing SGMM model, 
# Then we compute log-likelihoods and priors of the data given the model, and then we sum them up to get
# the posteriors. All outputs (alignments, priors, log-likelihoods, posteriors) will be in <output-dir>. 

set -x
# Begin configuration section.
cmd="queue.pl -l arch=*64"
stage=0
resplit_nj=15

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# -le 2 ]; then
  echo "Usage: $0 <input-pdf-post-dir1>[:weight1] <input-pdf-post-dir2>[:weight2] ... <output-pdf-post-dir>"
  echo "e.g.:  steps/post/combine_pdf_post.sh --transform-dir exp/tri3b exp/sgmm2_post:1 exp/nnet2_post:1 exp/combined_post."
  exit 1;
fi
combined_postdir=${@: -1}  # last argument to the script
post_dirs=( $@ )  # read the remaining arguments into an array
unset post_dirs[${#post_dirs[@]}-1]  # 'pop' the last argument which is odir
num_post=${#post_dirs[@]}  # number of systems to combine

mkdir -p $combined_postdir
total_sum=0
for i in `seq 0 $[num_post-1]`; do
  post_dir=${post_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  [ -z "$offset" ] && offset=1
  total_sum=$(($total_sum+$offset))
done

for i in `seq 0 $[num_post-1]`; do
  post_dir=${post_dirs[$i]}
  offset=`echo $post_dir | cut -d: -s -f2` # add this to the lm-weight.
  post_dir=`echo $post_dir | cut -d: -f1`
  [ -z "$offset" ] && offset=1
  weight=$(perl -e "print ($offset/$total_sum);")

  if [ ! -f $post_dir/post.1.scp ] && [ -f $post_dir/ali.1.gz ]; then # convert alignments to posteriors and put them in the same dir.
    nj=`cat $post_dir/num_jobs`
    $cmd JOB=1:$nj $post_dir/log/convert_ali_to_post.JOB.log \
      ali-to-pdf $post_dir/final.mdl "ark,s,cs:gunzip -c $post_dir/ali.JOB.gz|" ark:- \| \
      ali-to-post ark:- ark,scp:$post_dir/post.JOB.ark,$post_dir/post.JOB.scp || exit 1
  fi
  if [ -f $post_dir/post.1.scp ] ; then
    # Here, we re-split the posterior scp file, in order to speed up the combination process.
    cat $post_dir/post.*.scp | sort -k 1 > $post_dir/all_post.scp
    rm -rf $post_dir/resplit${resplit_nj}
    mkdir -p $post_dir/resplit${resplit_nj}
    resplit_scp=""
    for j in `seq 1 $resplit_nj`; do
      resplit_scp="$resplit_scp $post_dir/resplit${resplit_nj}/post.$j.scp"
    done  
    utils/split_scp.pl $post_dir/all_post.scp $resplit_scp
    post_split_dir=$post_dir/resplit${resplit_nj}

    # Here we combine the averaged posteriors with one more system during each iteration. 
    echo "Combining posteriors (${i}th iteration )..."
    if [ $i -eq 0 ]; then
      $cmd JOB=1:$resplit_nj $combined_postdir/log/combine_post_${i}.JOB.log \
        scale-post scp:$post_split_dir/post.JOB.scp $weight ark,scp:$combined_postdir/post_tmp_${i}.JOB.ark,$combined_postdir/post_tmp_${i}.JOB.scp
      post_dir_prev=$post_dir
    else
      echo $post_dir_prev
      prev_num_leaves=`tree-info $post_dir_prev/tree 2>/dev/null | awk '/num-pdfs/{print $NF}'` || exit 1;
      curr_num_leaves=`tree-info $post_dir/tree 2>/dev/null | awk '/num-pdfs/{print $NF}'` || exit 1;
      ! [ $prev_num_leaves -eq $curr_num_leaves ] \
        && echo "Cannot combine posteriors because num_pdfs in the systems don't match ($prev_num_leaves vs $curr_num_leaves)." && exit 1
      $cmd JOB=1:$resplit_nj $combined_postdir/log/combine_post_${i}.JOB.log \
        sum-post --scale1=$weight scp:$post_split_dir/post.JOB.scp scp:$combined_postdir/post_tmp_$[i-1].JOB.scp \
        ark,scp:$combined_postdir/post_tmp_${i}.JOB.ark,$combined_postdir/post_tmp_${i}.JOB.scp || exit 1
    fi

    # At the last iteration, we finish the posterior combination and create the final scp file.
    if [ $i -eq $[num_post-1] ]; then
      cat $combined_postdir/post_tmp_${i}.*.scp | sort -k 1 > $combined_postdir/post_combined.scp
      rm -rf $combined_postdir/post_tmp_$[i].*.scp
      cp $post_dir/{final.mdl,tree} $combined_postdir
    fi
    rm -rf $combined_postdir/post_tmp_$[i-1].*
  else
    echo "Posterior/Alignment does not exist!" && exit 1
  fi
done
exit

