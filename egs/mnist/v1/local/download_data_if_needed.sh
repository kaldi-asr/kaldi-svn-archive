#!/bin/bash

# Copyright 2014  John Hopkins University (author: Daniel Povey)
# Apache 2.0.



if [ $# -ne 1 ]; then
  echo "Usage: $0 <destination-path>"
  echo "e.g.: $0 /export/corpora5/NIST/MNIST"
  echo "This script will download the data from http://yann.lecun.com/exdb/mnist/ only if"
  echo "it does not already exist at that location."
  exit 1;
fi

url=http://yann.lecun.com/exdb/mnist/

dir=$1;

if [ ! -d $dir ]; then
  echo "$0: destination directory $dir does not exist, please make it."
  exit 1;
fi
if ! which curl >/dev/null; then
  echo "$0: the program 'curl' does not seem to be installed."
  exit 1;
fi

for file in train-images-idx3-ubyte.gz \
 train-labels-idx1-ubyte.gz \
 t10k-images-idx3-ubyte.gz \
 t10k-labels-idx1-ubyte.gz; do 
  if [ ! -s $dir/$file ]; then
    echo "$0: Downloading file $file"
    if [ ! -w $dir ]; then
      echo "$0: destination directory $dir is not writable."
      exit 1;
    fi
    if ! curl $url/$file > $dir/$file; then
      echo "$0: error downloading file $url/$file"
      exit 1;
    fi
  else 
    echo "$0: $dir/$file already exists."
  fi
done

echo "$0: success."
exit 0;
