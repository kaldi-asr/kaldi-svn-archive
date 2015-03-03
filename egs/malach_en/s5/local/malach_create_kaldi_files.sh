#!/bin/bash  

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.
# Begin configuration section.
map_oov=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -e
set -o pipefail
set -u

src=$1
lang=$2
dest=$3

mkdir -p $dest
transcript=$src/transcriptions.txt
if $map_oov ; then
  cat  $transcript | \
    local/map_oov.pl --symbol "<UNK>"  $lang/words.txt \
    > $src/transcriptions_mapped.txt
fi

local/malach_create_kaldi_files.pl $src/audio.lst \
  $transcript  $dest  2>$dest/conversion.log

utils/fix_data_dir.sh $dest

rm -rf $dest/reco2file_and_channel
cat $dest/segments | perl -ane '{$total_sec += $F[3] - $F[2];  } 
  END{ print "Extracted totally " . $total_sec/3600.00 . "hours in '$dest' \n";}';

