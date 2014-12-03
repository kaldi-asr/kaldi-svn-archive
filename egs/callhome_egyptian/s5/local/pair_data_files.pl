#!/usr/bin/env perl

use utf8;
use warnings;
use strict;
use Getopt::Long;
use Data::Dumper;

my $dump_all=0;
GetOptions ("all-matches" => \$dump_all);

open(AUDIO, "<$ARGV[0]");
open(TEXTS, "<$ARGV[1]");

my $target = $ARGV[2];

my %AUDMAP;
while (my $file = <AUDIO> ) {
  chomp $file;
  my $rec_name= `basename $file`;
  chomp $rec_name;
  my $key = uc($rec_name);
  $key =~ s/\..*//g;

  if ( exists $AUDMAP{$key} ) {
    push @{$AUDMAP{$key}}, $file;
  } else {
    push @{$AUDMAP{$key}}, $file;
  }
}

my %TEXTMAP;
while (my $file = <TEXTS> ) {
  chomp $file;
  my $rec_name= `basename $file`;
  chomp $rec_name;
  my $key = uc($rec_name);
  $key =~ s/\..*//g;
  if ( exists $TEXTMAP{$key} ) {
    push @{$TEXTMAP{$key}}, $file;
  } else {
    push @{$TEXTMAP{$key}}, $file;
  }
}

print "AUdio files found   : " . scalar(keys %AUDMAP) . "\n";
print "Transcriptions found: " . scalar(keys %TEXTMAP) . "\n";

#This is slightly tricky part -- we have to choose between several
#alternative annotations and select the best one
#as a rule of thumb, we should always prefer .su.xml over .scr
#and we should always prefer later BOLT release over the previous
my %LDC_CORPORA_WEIGHTS =  (
  "LDC2014R64" => 9, 
  "LDC2014E103" => 8, 
  "LDC2014E86" => 7, 
  "LDC2014E79" => 6, 
  "LDC2014E70" => 5, 
  "LDC2014E39" => 4,
  "LDC97T19" => 1, 
  "LDC2002T39" => 1, 
  "LDC2002T38" => 1,
  );

if (not $dump_all ) {
  print "Resolving the duplicities in annotations\n";
  for my $entry (keys %TEXTMAP) {
    next if scalar @{$TEXTMAP{$entry}} == 1;
    
    #print "$entry: " .  Dumper($TEXTMAP{$entry});

    my $weight = 0;
    my $used_path = "";
    my $used_ldc = "";
    for my $path (@{$TEXTMAP{$entry}}) {
      my $ldc_corpus=$path;
      $ldc_corpus=~s/.*(LDC[0-9A-Z]+).*/$1/;
      
      die "LDC corpus ID could not be isolated or is unknown to the script: $ldc_corpus" unless exists $LDC_CORPORA_WEIGHTS{$ldc_corpus};

      if ($weight < $LDC_CORPORA_WEIGHTS{$ldc_corpus} ) {
        $weight = $LDC_CORPORA_WEIGHTS{$ldc_corpus};
        $used_path = $path;
        $used_ldc = $ldc_corpus;
      } elsif ($weight < $LDC_CORPORA_WEIGHTS{$ldc_corpus} ) {
        die "Weight ambiguity between $ldc_corpus and $used_ldc! That should not happen!";
      }
    }
    #print "For key $entry and paths: \n";
    #print Dumper($TEXTMAP{$entry});
    print "Selected (with weight $weight) $used_path\n";
    delete $TEXTMAP{$entry};
    push @{$TEXTMAP{$entry}}, $used_path;
  }
}

print "Checking for duplicities in audio (shouldn't be any)\n";
for my $entry (keys %AUDMAP) {
  die "Duplicity in audio filenames found: " . Dumper($AUDMAP{$entry}) if scalar @{$AUDMAP{$entry}} != 1;
}
print "--> OK\n";

print "Matching audio vs annotation\n";
my @common_keys;
for my $key (keys %AUDMAP) {
  if (exists $TEXTMAP{$key} ) {
    push @common_keys, $key;
  }
}
print "--> Found " . scalar(@common_keys) . " common keys in total\n";

print "Writing map (audio, annotation) into $target/map.txt\n";
open(MAP, ">", "$target/map.txt") or die "Could not open $target/map.txt: $!";
for my $key (sort @common_keys) {
  die "Only one shall live!" if scalar(@{$AUDMAP{$key}}) != 1 ;
  die "Only one shall live!" if scalar(@{$TEXTMAP{$key}}) != 1 and not $dump_all;

  my $audio=@{$AUDMAP{$key}}[0];
  foreach my $text (@{$TEXTMAP{$key}}) {
    print MAP $audio . "\t" . $text . "\n";
  }
}
close(MAP);
