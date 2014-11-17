#!/bin/bash

cat -  | \
  sed 's/[A-Z][A-Z]*_LDC2014E86_ar_/ar_/g' |\
  sed 's/^\([a-z][a-z_0-9]*\)_[AB][0-9]* /\1 /g' | \
  perl -ane 'if ($_ =~ /;/) {print $_; next;} 
             @F=split;
             die "Unknown format of the line $_" unless @F >= 5;

             print "$F[0] $F[1] $F[2] $F[3] $F[4]";
             @words=();
             foreach $w(@F[5..$#F]) { 
              if ( $w =~ /.*_bw/) {
                push @words, $w;
              } elsif ( $w =~ /%/ ) {
                push @words, $w;
              } else {
                if ( $w =~ /[a-zA-Z]+/ ) {
                  push @words, uc(${w}) . "_en";
                } else {
                  push @words, ${w};
                }
              }
            }
            print " " . join(" ", @words) . "\n";
            ' 

