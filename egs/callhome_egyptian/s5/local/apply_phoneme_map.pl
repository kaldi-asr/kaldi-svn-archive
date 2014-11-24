#!/usr/bin/env perl

use strict;
use warnings;
use diagnostics;

use Data::Dumper;
open(MAP, $ARGV[0]) or die "Cannot open the char map: $!";
my %MAPPING;
while (my $line=<MAP>) {
  chomp $line;
  (my $phone, my $mappings) = split " ", $line, 2;
  #print STDERR $phone . "\n";
  #print STDERR $mappings . "\n";
  my @mapseq = split /,/, $mappings;
  #print STDERR Dumper(\@mapseq);
  push @{$MAPPING{$phone}}, @mapseq;
  #print STDERR Dumper(\%MAPPING);
}
close(MAP);
#print STDERR Dumper(\%MAPPING);
while ( my $line=<STDIN> ) {
  #print STDERR $line;
  chomp $line;
  (my $phone, my $pron_str) = split " ", $line, 2;
  if ( not defined($pron_str)  ) {
    die "Cannot parse \"$line\"\n";
  }

  my @pron = split " ", $pron_str;
  my @out_prons = ("") ;
  foreach $phone (@pron) {
    #print STDERR Dumper(\@out_prons, \@pron, $pron_str);
    if (not defined($MAPPING{$phone})) {
      die "Undefined phoneme \"$phone\"\n";
    }
    my @replacements = @{$MAPPING{$phone}};
    my @tmp;
    foreach my $repl (@replacements) {
      foreach my $pos (@out_prons) {
        push @tmp, "$pos $repl";
      }
    }
    @out_prons= @tmp;
    @tmp = ();
  }
  #print STDERR Dumper(\@out_prons);

  foreach  my $pron_var (@out_prons) {
    my $s = "$phone\t$pron_var";
    $s =~ s/^\s+|\s+$//g;
    print $s . "\n";
  }
}

