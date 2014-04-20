#!/bin/bash


# This is somewhere to put the data; it can just be a local path if you
# don't want to give it a permanent home.
data=/export/corpora5/NIST/MNIST

local/download_data_if_needed.sh $data || exit 1;

local/format_data.sh $data


