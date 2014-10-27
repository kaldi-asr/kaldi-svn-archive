#!/usr/bin/env perl 
use warnings;
use strict;
use utf8;
use Data::Dumper;
use Getopt::Long;

binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";
binmode STDIN, ":utf8";

my $SYSID="BOLT_ARZ_CTS_JHU";
GetOptions ( "sysid=s"   => \$SYSID);

my $filelist=$ARGV[0];
my $datadir=$ARGV[1];

my %WAVS;
open (my $WAV_FILE, "$datadir/wav.scp") or die "Cannot open file $datadir/wav.scp: $!";
while (my $line = <$WAV_FILE> ) {
  chomp $line;
  my @entries=split(" ", $line, 2);
  die "Cannot parse wav.scp line $line (no of entries != 2)" if @entries != 2;
  push @{$WAVS{$entries[0]}}, @entries;
}
close($WAV_FILE);
#print Dumper(\%WAVS);

my %SEGMENTS;
open (my $SEGMENT_FILE, "$datadir/segments") or die "Canot open file $datadir/segments: $!";
while (my $line=<$SEGMENT_FILE>) {
  chomp $line;
  my @entries=split(" ", $line);
  die "Cannot parse segment line $line (no of entries != 4)" if @entries != 4;

  push @{$SEGMENTS{$entries[0]}}, @entries;
}
close($SEGMENT_FILE);
#print Dumper(\%SEGMENTS);

my %FILES;
open( my $FILES_FILE, $filelist);
while (my $line=<$FILES_FILE>) {
  chomp $line;

  my @entries=split(" ", $line);
  die "Cannot parse $filelist line $line (no of entries != 2)" if @entries != 2;
  push @{$FILES{$entries[1]}}, @entries;
}
close( $FILES_FILE);
#print Dumper(\%FILES);

my %CTM;
while (my $line=<STDIN>) {
  chomp $line;

  my @entries=split(" ", $line);
  die "Cannot parse CTM (\@STDIN) $line (no of entries != 6)" if @entries != 6;
  #my $arr= $CTM{$entries[0]};
  #push @{${$arr}{$entries[1]}},  $entries[4];
  #$CTM{$entries[0]}= $arr;
  push @{$CTM{$entries[0]}}, $entries[4];
}
#print Dumper(\%CTM);

foreach my $file_xml (keys %FILES) {
  open(my $XML_FILE, "<:encoding(utf-8)", $file_xml);
  my $conversation=undef;
  my $rec_id=undef;
  while (my $line=<$XML_FILE>) {
    chomp $line;
   
    if ( $line =~ s/<conversation *id="(.*)">/$1/g ) {
      $conversation=$line;
      $rec_id=uc($conversation);
      print "<DOC docid=\"$conversation\" sysid=\"$SYSID\">\n";
    } elsif ( $line =~ m/<su/ ) {
      die "Element <su> read, but <conversation> didn't started!" unless defined $conversation;
      my $id=$line;
      $id =~ s/.* id="(.*?)" .*/$1/g;
      my $speaker=$line;
      $speaker =~ s/.*speaker="(.*?)".*/$1/g;
      my $channel=$line;
      $channel =~ s/.*channel="(.*?)".*/$1/g;
      my $begin=$line;
      $begin =~ s/.*begin="(.*?)".*/$1/g;
      my $end=$line;
      $end =~ s/.*end="(.*?)".*/$1/g;
      

      my $utt_id=sprintf "%s-%s-%06.0f-%06.0f", $rec_id, $speaker, 100*$begin, 100*$end;
      my $text="";
      $text = join(" ", @{$CTM{$utt_id}}) if defined $CTM{$utt_id};
      #print Dumper($utt_id, $CTM{$utt_id});
      print "  <seg id=\"$id\" speaker=\"$speaker\" channel=\"$channel\" begin=\"$begin\" end=\"$end\"> $text </seg>\n";
    }
  }
  print "</DOC>\n" if (defined $conversation); 
  close($XML_FILE);
}
