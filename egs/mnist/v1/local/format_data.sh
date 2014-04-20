#!/bin/bash

# Copyright 2014  John Hopkins University (author: Daniel Povey)
# Apache 2.0.


if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-location>"
  echo "e.g.: $0 /export/corpora5/NIST/MNIST"
  echo "This script will take the data from http://yann.lecun.com/exdb/mnist/"
  echo "and format it in a more Kaldi-like format in data/train and data/test."
  exit 1;
fi

. ./path.sh || exit 1;

srcdir=$1

temp=images/ # where we put the archive data.

mkdir -p $temp || exit 1;

for file in train-images-idx3-ubyte.gz \
 train-labels-idx1-ubyte.gz \
 t10k-images-idx3-ubyte.gz \
 t10k-labels-idx1-ubyte.gz; do 
  if [ ! -f $srcdir/$file ]; then
    echo "$0: expected file $srcdir/$file to exist."
    exit 1;
  fi
done

for type in t10k train; do
  mkdir -p data/$type || exit 1;

  echo "$0: Creating class labels for $type";

  cat <<EOF | python -u - <(gunzip -c $srcdir/${type}-labels-idx1-ubyte.gz) > data/$type/labels
import sys, struct;
if (len(sys.argv) != 2):
  print '%s: expected a single command line argument.' % sys.argv[0]
  exit(1)

filename = sys.argv[1];
f = open(filename, "rb");

magic_number = struct.unpack('>I', f.read(4))[0] # MNIST data is big-endian 
if magic_number != 2049:
  sys.exit("Wrong magic number detected: %d" % magic_number);
num_images = struct.unpack('>I', f.read(4))[0]

if num_images == 60000:
  subset_name = 'train'
else:
  subset_name = 't10k'

for n in range(num_images):
  label = struct.unpack('B', f.read(1))[0]
  print "mnist_%s_%d %d" % (subset_name, n, label)
EOF
done


for type in t10k train; do
  mkdir -p data/$type || exit 1;

  echo "$0: Creating archive of image data for $type";

  cat <<EOF | python -u - <(gunzip -c $srcdir/${type}-images-idx3-ubyte.gz) | copy-feats  \
     --compress=true ark:- ark,scp:$PWD/images/$type.ark,$PWD/images/$type.scp || exit 1;
import sys, struct;
if (len(sys.argv) != 2):
  print '%s: expected a single command line argument.' % sys.argv[0]
  exit(1)

filename = sys.argv[1];
f = open(filename, "rb");

magic_number = struct.unpack('>I', f.read(4))[0] # MNIST data is big-endian 
if magic_number != 2051:
  sys.exit("Wrong magic number detected: %d" % magic_number);
num_images = struct.unpack('>I', f.read(4))[0]
num_rows = struct.unpack('>I', f.read(4))[0]
num_cols = struct.unpack('>I', f.read(4))[0]
if num_rows != 28 or num_cols != 28:
  sys.exit("Number of rows or columns is wrong: %d, %d", num_rows, num_cols);

if num_images == 60000:
  subset_name = 'train'
else:
  subset_name = 't10k'

for n in range(num_images):
  print "mnist_%s_%d [" % (subset_name, n),
  for r in range(num_rows):
    data=f.read(num_cols);
    if len(data) != num_cols:
      sys.exit('Error reading data (truncated file?)');
    format_str = '%dB' % num_cols; # e.g. '28B'
    row = struct.unpack(format_str, data);  # get it as 28 ints.
    for pixel in row:
      print (pixel/256.0), ' ',
    print;  # newline.
  print "]";
EOF

cp $PWD/images/$type.scp  data/$type/feats.scp

done

