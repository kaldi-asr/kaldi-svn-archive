#!/usr/bin/env perl

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

use strict;
use warnings;
use XML::Parser;
use Data::Dumper;
use Getopt::Long;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $SYMBOL="<OOV>";
GetOptions ("symbol=s" => \$SYMBOL);

my %DICT;
print STDERR "$0: Reading the lexicon\n";
open( my $DIC, "<:utf8", $ARGV[0]) or die "Could not open lexicon \"$ARGV[0]\": $!\n";
while (my $line = <$DIC>) {
  chomp $line;
  (my $word, my $pron) = split(" ", $line, 2);
  $DICT{$word}=1;
}
close($DIC);

my $iv_words = 0;
my $oov_words = 0;

print STDERR "$0: Re-mapping the OOV words (replacing with $SYMBOL)\n";
while ( my $line = <STDIN> ) {
  my @words = split (" " , $line);
  for ( my $i = 1; $i <= $#words; $i+=1) {
    if ( $DICT{$words[$i]} ) {
      $iv_words +=1;
    } else {
      $words[$i] = $SYMBOL;
      $oov_words +=1;
    }
  }
  print join(" ", @words) . "\n";
}
my $words_total = $iv_words + $oov_words;
print STDERR "$0: Done, total $words_total words, $iv_words IV, $oov_words OOV\n";
