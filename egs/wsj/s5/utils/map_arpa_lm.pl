#!/usr/bin/perl

# Copyright 2014  Guoguo Chen
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
This script reads the Arpa format language model, and maps the words into
integers or vice versa. It ignores the words that are not in the symbol table,
and updates the head information.

It will be used joinly with lmbin/arpa-to-const-arpa to build ConstArpaLm format
language model. We first map the words in an Arpa format language model to
integers, and then use lmbin/arpa-to-const-arpa to build a ConstArpaLm format
language model.

Usage: utils/map_arpa_lm.pl [options] <words.txt> <input-arpa> <output-arpa>
 e.g.: utils/map_arpa_lm.pl words.txt arpa_lm.txt arpa_lm.int

Allowed options:
  --sym2int   : If true, maps words to integers, other wise maps integers to
                words. (boolean, default = true)

EOU

my $sym2int = "true";
GetOptions('sym2int=s' => \$sym2int);

($sym2int eq "true" || $sym2int eq "false") ||
  die "$0: Bad value for option --sym2int\n";

if (@ARGV != 3) {
  die $Usage;
}

# Gets parameters.
my $symtab = shift @ARGV;
my $arpa_in = shift @ARGV;
my $arpa_out = shift @ARGV;

# Opens files.
open(M, "<$symtab") || die "$0: Fail to open $symtab\n";
open(I, "<$arpa_in") || die "$0: Fail to open $arpa_in\n";
open(O, ">$arpa_out") || die "$0: Fail to open $arpa_out\n";

# Reads in the mapper.
my %mapper;
while (<M>) {
  chomp;
  my @col = split(/[\s]+/, $_);
  @col == 2 || die "$0: Bad line in mapper file \"$_\"\n";
  if ($sym2int eq "true") {
    if (defined($mapper{$col[0]})) {
      die "$0: Duplicate entry \"$col[0]\"\n";
    }
    $mapper{$col[0]} = $col[1];
  } else {
    if (defined($mapper{$col[1]})) {
      die "$0: Duplicate entry \"$col[1]\"\n";
    }
    $mapper{$col[1]} = $col[0];
  }
}

# Parses Arpa n-gram language model.
my $arpa = "";
my $current_order = -1;
my %head_ngram_count;
my %actual_ngram_count;
while (<I>) {
  chomp;
  my @col = split(/[\s]+/, $_);

  if (m/^\\data\\$/) {
    $current_order = 0;
  } elsif (m/^\\[0-9]*-grams:$/) {
    $current_order = $_;
    $current_order =~ s/-grams:$//g;
    $current_order =~ s/^\\//g;
    $arpa .= "$_\n";
  } elsif (m/\\end\\/) {
    $arpa .= "$_\n";
  } elsif ($_ eq "") {
    if ($current_order >= 1) {
      $arpa .= "\n";
    }
  } else {
    if ($current_order == 0) {
      # Parses head section.
      if ($col[0] ne "ngram" || @col != 2) {
        die "$0: Expecting \"ngram\" token in head section, got \"$_\"\n";
      } else {
        my @sub_col = split("=", $col[1]);
        @sub_col == 2 || die "$0: Bad line in arpa lm \"$_\"\n";
        $head_ngram_count{$sub_col[0]} = $sub_col[1];
      }
    } else {
      # Parses n-gram section.
      if (@col > 2 + $current_order || @col < 1 + $current_order) {
        die "$0: Bad line in arpa lm \"$_\"\n";
      }
      if (!defined($actual_ngram_count{$current_order})) {
        $actual_ngram_count{$current_order} = 0;
      }
      my $new_line = "$col[0]\t";
      my $is_oov = "false";
      for (my $i = 1; $i <= $current_order; $i++) {
        if (!defined($mapper{$col[$i]})) {
          $is_oov = "true";
          last;
        }
        $new_line .= $mapper{$col[$i]};
        if ($i != $current_order) {
          $new_line .= " ";
        }
      }
      if ($is_oov eq "false") {
        if (@col == 2 + $current_order) {
          $new_line .= "\t$col[1 + $current_order]";
        }
        $arpa .= "$new_line\n";
        $actual_ngram_count{$current_order} += 1;
      }
    }
  }
}

foreach my $order (keys(%head_ngram_count)) {
  if ($head_ngram_count{$order} < $actual_ngram_count{$order}) {
    die "$0: Expecting $head_ngram_count{$order} $order-grams, seeing more.\n"
  }
}

my $header = "\n\\data\\\n";
foreach my $order (sort(keys(%actual_ngram_count))) {
  $header .= "ngram $order=$actual_ngram_count{$order}\n";
}
$arpa = $header . "\n" . $arpa;

print O $arpa;

close(M);
close(I);
close(O);
