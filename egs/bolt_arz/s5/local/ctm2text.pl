#! /usr/bin/env perl
use Data::Dumper;
use diagnostics;

my @utt_ids;
my %utt;
open(SEGS, $ARGV[0]);
while( <SEGS> ) {
  chomp;
  @F = split " ";
  push @utt_ids, $F[0];
}
close(SEGS);

while ( <STDIN> ) {
  chomp;
  @F = split ;
  $channel=$F[1];
  
  die "Unsupported format of the ctm.utt: channel is not 1" unless $channel eq 1;
  push @{$utt{$F[0]}->{$channel}}, $F[4];
}

#print Dumper(\%utt)
#print Dumper(\@utt_ids);

foreach $id (@utt_ids) {
  if (exists $utt{$id}->{1} ) {
    print "$id " . join(" ", @{$utt{$id}->{1}}) . "\n";
  } else {
    print "$id\n";
  }
}
