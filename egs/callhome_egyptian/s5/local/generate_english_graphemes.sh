#!/bin/bash

g2g=$1
words=$2
lexicon=$3

. ./cmd.sh
. ./path.sh


local/apply_g2p.sh --with-probs false --icu-transform "" \
  --cmd "$train_cmd" --output-lex $g2g/lexicon.tmp \
  <( sed 's/_en$//g' $words | sort -u )  $g2g $g2g


cat $g2g/lexicon.tmp | perl -e '
  while( $line = <STDIN> ) {
    chomp $line;
    @recs = split "\t", $line;
    die "Cannot parse line $line " if @recs !=2;
    $word=$recs[0];
    $pron=$recs[1];
    print "${word}_en\t$pron\n";
  }
' > $lexicon
