#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.
seg_length=30
min_seg_length=10
overlap_length=5
# End configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <input-dir> <output-dir>"
  echo "Options:"
  echo "    --min-seg-length        # minimal segment length"
  echo "    --seg-length            # length of segments in seconds."
  echo "    --overlap-length        # length of overlap in seconds."
  exit 1;
fi

input_dir=$1
output_dir=$2

for f in spk2utt text utt2spk wav.scp; do
  [ ! -f $input_dir/$f ] && echo "$0: no such file $input_dir/$f" && exit 1;
done

[ ! $seg_length -gt $overlap_length ] \
  && echo "$0: --seg-length should be longer than --overlap-length." && exit 1;


mkdir -p $output_dir
cp -f $input_dir/spk2gender $output_dir/spk2gender 2>/dev/null
cp -f $input_dir/text $output_dir/text.orig
cp -f $input_dir/segments $output_dir/segments.orig
cp -f $input_dir/wav.scp $output_dir/wav.scp.orig

#steps/cleanup/split_long_utterances.pl --min-seg-length $min_seg_length \
#  --seg-length $seg_length --segment-overlap $overlap_length \
#  $input_dir $output_dir
steps/cleanup/split_long_utterances.pl  $input_dir $output_dir

utils/fix_data_dir.sh $output_dir

exit 0;
