#!/usr/bin/env perl

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

use strict;
use warnings;
use Data::Dumper;
use Getopt::Long;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

sub cartesian_str {
    my @C = [];

    foreach (reverse @_) {
        my @A = @$_;

        @C = map { my $n = $_; map {  [ "$n @$_" ]  } @C } @A;
    }

    my @a = map {@$_} @C;  
    return @a;
}

sub cartesiaan {
    my @C = map { [ $_ ] } @{ shift @_ };

    foreach (@_) {
        my @A = @$_;

        @C = map { my $n = $_; map { [ $n, @$_ ] } @C } @A;
    }

    return @C;
}

sub print_lexicon_entries {
  my $F = $_[0];
  my $W = $_[1];
  my @P = @{$_[2]};

  for my $p (@P) {
    print $F "$W\t$p\n";
  }

}
sub remove_brackets_and_look {

  my $LEX=$_[0];
  my $lookup = $_[1];

  #knowledge-based system
  $lookup=~s/'//g;
  #$lookup=~s/\((.*)\)/\1/g;
  $lookup=~s/\(//g;
  $lookup=~s/\)//g;
  $lookup=~s/\[//g;
  $lookup=~s/\[//g;

  if (exists $LEX->{$lookup} ) {
    return $LEX->{$lookup};
  } else {
    return ;
  }
}

sub split_and_look {
  my $LEX=$_[0];
  my $lookup = $_[1];

  return if $lookup =~ m/ -|-$/; 

  my @split_words = split("-", $lookup);
  my @split_prons;
  foreach my $word (@split_words) {
    if ( $LEX->{$word} ) {
      push @split_prons, $LEX->{$word};
    } else {
      return
    }
  }
  #print STDERR "Split_prons{$lookup): \n" . Dumper(\@split_prons) . "\n";
  my @prons = cartesian_str(@split_prons);
  #print STDERR "Combined_prons($lookup):\n" .  Dumper(\@prons) . "\n";
  return @prons;
}

my $cmudict=$ARGV[0];
my $wordlist=$ARGV[1];

my $lexout=$ARGV[2];
my $oov=$ARGV[3];

open(CMU, "<:utf8", $cmudict);
open(WORDS, "<:utf8", $wordlist);
open(LEXICON, ">:utf8", $lexout);
open(OOV, ">:utf8", $oov);
my %LEX;

while (my $line = <CMU>) {
  chomp $line;
  next if $line=~ /^;;;/;
  (my $word, my $pron) = split(" ", $line, 2);
  $word=~s/\([0-9]\)$//;
  $pron=~s/[0-9]//g;
  
  push @{$LEX{$word}}, $pron;
}
close(CMU);

my $total_words=0;
my $found_words=0;
my $oov_words=0;

my $total_tokens=0;
my $found_tokens=0;
my $oov_tokens=0;

while (my $line = <WORDS>) {
  chomp $line;
  $line =~ s/^\s+|\s+$//g;
  next if not $line;

  (my $wc, my $word) = split(" ", $line, 2);
  my $lookup = uc($word);
  $lookup =~ s/\@//g;
  
  if (exists $LEX{$lookup} ) {
    $found_words+=1;
    $found_tokens+=$wc;
    
    print_lexicon_entries(\*LEXICON, $word,  $LEX{$lookup});
  } elsif ($word =~ /<.*>/ ) {
    next;
  } elsif ( my @l = remove_brackets_and_look(\%LEX, $lookup) ) {
    $found_words+=1;
    $found_tokens+=$wc;
    print "REMOVE: " .  Dumper(\@l); 
    print_lexicon_entries(\*LEXICON, $word,  @l);
  } elsif ( my @m=split_and_look(\%LEX, $lookup) ) {
    #print Dumper(\@l);
    $found_words+=1;
    $found_tokens+=$wc;
    #print "SPLIT_LOOK: " . Dumper(\@m); 
    print_lexicon_entries(\*LEXICON, $word,  \@m);
  } else {
    $oov_words +=1;
    $oov_tokens += $wc;
    print OOV "$wc $word\n" ;
  }
  $total_words += 1;
  $total_tokens += $wc;

}
close(WORDS);
close(LEXICON);
close(OOV);

print "Parsed whole $wordlist: \n";
print "\tfound $found_words out of $total_words ($oov_words OOV)\n";
print "\tfound $found_tokens out of $total_tokens ($oov_tokens OOV)\n";
