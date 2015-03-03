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

my %SCORES;
my @HEADER;

while (my $line =  <STDIN> ) {
  my @entries = split(" ", $line);
  if ( $entries[1] eq "id") {
    @HEADER=@entries;
  }
  next if $entries[1] ne "raw";
  
  my $chan = $entries[0];
  my $new_chan = $chan;

  next if $chan !~ /-[AB]-/;

  $new_chan =~ s/-[AB]-/-*-/gi;
  #print Dumper( [$chan, $new_chan] );

  if (not defined $SCORES{$new_chan} ) {
    @{$SCORES{$new_chan}} = @entries;
  } else {
    @{$SCORES{$new_chan}} = @entries if $SCORES{$new_chan}->[8] > $entries[8];
  }
}

my $SENT = 0;
my $WORD = 0;
my $Corr = 0;
my $Sub  = 0;
my $Ins  = 0;
my $Del  = 0;
my $Err  = 0; 
my $SErr = 0;

print join(" ", @HEADER) . "\n";
foreach my $spk( sort(keys(%SCORES)) ) {
  print join(" ", @{$SCORES{$spk}}) . "\n";
  
  $SENT += $SCORES{$spk}->[2];
  $WORD += $SCORES{$spk}->[3];
  $Corr += $SCORES{$spk}->[4];
  $Sub  += $SCORES{$spk}->[5];
  $Ins += $SCORES{$spk}->[6];
  $Del += $SCORES{$spk}->[7];
  $Err += $SCORES{$spk}->[8];
  $SErr += $SCORES{$spk}->[9];
}

print "SUM raw $SENT $WORD $Corr $Sub $Ins $Del $Err $SErr\n";

my $x_corr = sprintf "%.2f", 100.0 * $Corr / $WORD;
my $x_sub = sprintf "%.2f", 100.0 * $Sub / $WORD;
my $x_ins = sprintf "%.2f", 100 * $Ins / $WORD;
my $x_del = sprintf "%.2f", 100 * $Del / $WORD;
my $x_err = sprintf "%.2f", 100 * $Err / $WORD;
my $x_serr = sprintf "%.2f", 100 * $SErr / $SENT;
print "SUM sys $SENT $WORD $x_corr $x_sub $x_ins $x_del $x_err $x_serr\n";
