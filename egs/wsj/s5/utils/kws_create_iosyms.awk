#!/usr/bin/awk -f

# Copyright 2012  Lucas Ondel (Brno University of Technology)
# Apache 2.0


# Create a reverted index of all utterances of the lattices provided. The
# resulting index is a WFST itself.

BEGIN {
  if (ARGC != 3) {
    print "usage: ./kws_create_iosyms.awk <input-symbol-table> <utterance-list>" > /dev/stderr;
    exit 1;
  }

  isyms = ARGV[1];
  utts = ARGV[2];
  id=0;
   
  # Copy the input symbol table
  while (getline < isyms) {
    id=$2;
    print $0;
  }

  # Add symbol table
  while (getline < utts) {
      printf "%s %s\n", $1, ++id; 
  }
}

