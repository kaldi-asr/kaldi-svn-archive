#!/usr/bin/env perl
use utf8;
use warnings;
use strict;
use Data::Dumper;

my %DB;

open (DBFILE, $ARGV[0]);
my $annotdir=$ARGV[2];
my $audiodir=$ARGV[1];
my $output=$ARGV[3];
my $total_seconds=0.0;

while (my $line=<DBFILE>) {
  chomp $line;
  my @records=split " ", $line;
  die "Unknown format of the line $line\n" if @records != 6;
  
  my $textfile=$records[0];
  my $seqid=$records[1];
  my $audiofile=$records[2];
  my $channid=$records[3];
  my $begin=$records[4];
  my $end=$records[5];

  die "I can do only channel=0 for now!" if $channid != 0;
 
  push @{$DB{$textfile}->{$seqid}}, @records;
  #if ($textfile eq "CALLHOME_G-ar_0072-8") {
  #  print Dumper(\@records);
  #  push @{$DB{$textfile}->{$seqid}}, @records;
  #}
}

close(DBFILE);


open(my $WAV, "| sort -u >$output/wav.scp") or die "Cannot open the file $output/wav.scp: $!";
open(my $UTT2SPK, ">$output/utt2spk") or die "Cannot open the file $output/utt2spk: $!";
open(my $SEGMENTS, ">$output/segments") or die "Cannot open the file $output/segments: $!";
open(my $TEXTS, ">$output/text") or die "Cannot open the file $output/text: $!";
open(my $RECO, "| sort -u >$output/reco2file_and_channel") or die "Cannot open the file $output/reco2file_and_channel: $!";

my $sph2pipe=`which sph2pipe` or die "Could not find the sph2pipe binary: $!";
chomp $sph2pipe;
my $sox=`which sox` or die "Could not find the sox binary: $!";
chomp $sox;


for my $textfile (keys %DB) {
  my $filename="$annotdir/${textfile}.txt";
  if ( ! -e $filename )  {
    print STDERR "File $filename does not exist!\n";
    next;
  }
  my $spk_id=$textfile;
  #$spk_id =~ s/^CALLHOME_G-//i;
  $spk_id = uc($spk_id);

  open (my $annotfile, "<$filename") or die "Cannot open file $filename: $!";
  #print Dumper($textfile);
  while (my $line=<$annotfile> ) {
    chomp $line;
    #print Dumper($line);

    (my $seqid, my $text) = split(/\s+/, $line, 2);
    next unless exists $DB{$textfile}{$seqid};
    #print Dumper($seqid);
    #print Dumper($text);
    #print $seqid . " " . $text . "\n";
    
    my @records = @{$DB{$textfile}->{$seqid}};

    my $audiofile=$records[2];
    my $channid=$records[3];
    my $begin=$records[4];
    my $end=$records[5];
    
    my $audio_id=uc($audiofile);
    $audio_id =~ s/\.SPH//i;

    my $utt_id=sprintf("%s-%06d-%06d", uc($textfile),  $begin * 100, $end*100);
    my $audio="$audiodir/$audiofile";

    unless ( -f $audio) {
      print STDERR "File $audio does not exist!\n";
      next;
    }
    $total_seconds+=$end-$begin;
    print $TEXTS "$utt_id $text\n";
    print $SEGMENTS "$utt_id $audio_id $begin $end\n";
    print $UTT2SPK "$utt_id $spk_id\n";
    print $RECO "$audio_id $audio_id A\n";
    print $WAV "$audio_id $sph2pipe -f wav -p -c 1 $audio| sox -t wav - -t wav -r 8000 -|\n";

  }

  close($annotfile);
}
print "Extracted $total_seconds seconds of audio data (" . $total_seconds/3600 . " hours)\n";
