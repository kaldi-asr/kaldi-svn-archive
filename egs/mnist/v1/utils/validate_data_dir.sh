#!/bin/bash



if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  echo "e.g.: $0 data/train"
fi

data=$1

if [ ! -d $data ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

for f in feats.scp labels; do
  if [ ! -f $data/$f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
  if [ ! -s $data/$f ]; then
    echo "$0: empty file $f"
    exit 1;
  fi
done

tmpdir=$(mktemp -d);
trap 'rm -rf "$tmpdir"' EXIT HUP INT PIPE TERM

export LC_ALL=C

function check_sorted_and_uniq {
  ! awk '{print $1}' $1 | sort | uniq | cmp -s - <(awk '{print $1}' $1) && \
    echo "$0: file $1 is not in sorted order or has duplicates" && exit 1;
}

function partial_diff {
  diff $1 $2 | head -n 6
  echo "..."
  diff $1 $2 | tail -n 6
  n1=`cat $1 | wc -l`
  n2=`cat $2 | wc -l`
  echo "[Lengths are $1=$n1 versus $2=$n2]"
}

check_sorted_and_uniq $data/feats.scp

check_sorted_and_uniq $data/labels


cat $data/feats.scp | awk '{print $1;}' > $tmpdir/examples

cat $data/labels | awk '{print $1;}' > $tmpdir/examples.labels

if ! cmp $tmpdir/examples $tmpdir/examples.labels; then
  echo "$0: Error: in $data, utterance lists extracted from utt2spk and text"
  echo "$0: differ, partial diff is:"
  partial_diff $tmpdir/utts{,.txt}
  exit 1;
fi


echo "Successfully validated data-directory $data"
