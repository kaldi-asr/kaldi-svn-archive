#!/bin/bash

lexicon=data/local/dict/lexicon-arz.txt.notags
en_lex=data/local/dict/lexicon-en-oov.txt

out=exp/g2p/rev_arz
out2=exp/g2p/en_to_arz_graphemes


. ./cmd.sh
. ./path.sh

mkdir -p $out
mkdir -p $out2
paste  \
  <(cut -f 2  data/local/dict/lexicon-arz.txt.notags | sed 's/ //g') \
  <(cut -f 1  $lexicon | sed 's/./\0 /g' ) \
  > $out/train.txt


#local/train_g2p.sh --icu-transform "" --lexicon $out/train.txt $out $out

#local/apply_g2p.sh --with-probs false --icu-transform "" --cmd "$train_cmd" \
#  <(cut -f 2 $en_lex | sed 's/_1 *//g' | sed 's/ *//g' | sort -u )  $out $out


cat $out/lexicon.lex | perl -e '
  use Data::Dumper;
  open(LEX, $ARGV[0]);
  while($line=<LEX>) {
    chomp $line;
    @recs = split "\t", $line;
    die "Cannot parse line $line " if @recs !=2;
    $word=$recs[0];
    $word=~s/_en//g;
    $pron=$recs[1];
    $pron=~s/_1//g;
    $pron=~s/^ *//g;
    $pron=~s/  *//g;
    push @{$LEXICON{$pron}}, $word;
  } 
  #print Dumper(\%LEXICON);
  while( $line = <STDIN> ) {
    chomp $line;
    @recs = split "\t", $line;
    die "Cannot parse line $line " if @recs !=2;
    $word=$recs[0];
    $pron=$recs[1];
    for $en_word (@{$LEXICON{$word}}) {
      print "$en_word\t$pron\n";
    }
  }
' $en_lex |sort -u > $out2/train.txt

local/train_g2p.sh --icu-transform "" --lexicon $out2/train.txt $out2 $out2


