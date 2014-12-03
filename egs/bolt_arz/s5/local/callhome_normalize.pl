#!/usr/bin/perl -w
# Copyright  2014 Johns Hopkins University(Jan Trmal)


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.



# normalizations for CALLHOME trascript
# See the transcription documentation for details

use Data::Dumper;

@vocalized_noise=( "{breath}", "{cough}", "{sneeze}", "{lipsmack}" );

%tags=();
%noises=();
%unint=();
%langs=();

LINE: while (<STDIN>) {
  $line=$_;
  #print STDERR Dumper($line);
  #$line =~ s/\+/ /g;
  @A = split(" ", $line);
  print "$A[0] ";

  $n = 1;
  while ($n < @A ) {
    $a = $A[$n]; 
    $n+=1;
    #print STDERR Dumper($a);

    if ( grep { $_ =~ m/^\Q$a/i } @vocalized_noise ) {
      print "<V-NOISE> ";
      next;
    }

    if ( $a =~ /{laugh}/i ) {
      print "<LAUGHTER> ";
      next;
    }

    if ( $a =~/^\[\[/ ) {
      #[[text]]          comment; most often used to describe unusual
      #                  characteristics of immediately preceding or following
      #                  speech (as opposed to separate noise event)
      #                  [[previous word lengthened]]    [[speaker is singing]]
      $tags{$a} += 1;
      next;
    }

    if ( $a =~/^\{/ ) {
      #{text}              sound made by the talker
      #                    examples: {laugh} {cough} {sneeze} {breath}
      $noises{$a} += 1;
      print "<V-NOISE> ";
      next;
    }

    if ( $a =~/<NOISE>/i ) {
      #This is for the RT04F speech corpus
      $noises{$a} += 1;
      print "<NOISE> ";
      next;
    }

    if ( $a =~/^\[/ ) {
      #[text]              sound not made by the talker (background or channel)
      #                    examples:   [distortion]    [background noise]      [buzz]
      $noises{$a} += 1;
      print "<NOISE> ";
      next;
    }

    if ( $a =~/^\[\// ) {
      #[/text]             end of continuous or intermittent sound not made by
      #                    the talker (beginning marked with previous [text])
      $noises{$a} += 1;
      next;
    }

    if ( $a =~/^\(\(/ ) {
      #((text))            unintelligible; text is best guess at transcription
      #                    ((coffee klatch))
      #(( ))               unintelligible; can't even guess text
      #                    (( ))
      $tmp=$a;
      $b=$a;
      $b =~ s/^\(\(//g;
      print "<UNK> " if $b;

      while (( $a !~ /.*\)\)/ ) && ($n < @A)) {
        #print Dumper(\@A, $tmp, $a, $n);
        $a = $A[$n]; 
        $n += 1;
        $tmp = "$tmp $a";
        
        $b=$a;
        $b =~ s/^\(\(//g;
        $b =~ s/\)\)$//g;
        print "<UNK> " if $b;
      }
      $unint{$tmp} += 1;
      next;
    }
    if (( $a =~/^-[^-]*$/)   || 
        ( $a =~/^[^-]+-$/)   || 
        ( $a =~/^[^-]+.*-$/) || 
        ($a =~ /\*.*\*$/ ) ) {
      #text-   partial word
      #        example: absolu-
      #**text**    idiosyncratic word, not in common use, not necessarily 
      #            included in lexicon
      #            Example: **poodle-ish**

      print "<UNK> ";
      next;
    }

    if (( $a =~ /^\+.*?\+$/) || ( $a =~/^\@.*\@$/)) {
      #+text+  mandarinized foreign name, no standard spelling
      #@text@  a abbreviation word
      #print STDERR Dumper($a);
      $a =~ s/^[+@](.*)[+@]$/$1/;
      $a =~ s/^&//g; 
      $a =~ s/B\~/B/g;
      $a =~ s/B\(t\)/B/g;
      $a =~ s/il\(k\)/il/g;
      $a =~ s/il\(g\)/il/g;
      $a =~ s/(.+)\+&/$1+/g;
      print "$a ";
      next;
    }
    if (( $a =~/^%/ ) || ( $a =~/^!/ ))  {
      #%text         This symbol flags non-lexemes, which are
      #!text         This symbol flags non-lexemes, which are
      #              general hesitation sounds.
      #              Example: %mm %uh
      $noises{$a} += 1;
      print "<HES> ";
      next;
    }

    if (( $a =~/^;/ ) || ( $a =~/^\&/ ) || ($a eq "--") || ($a =~ /^\/\// ) || ($a =~ /\/\/$/ ))  {
      #;text      used to mark proper names and place names
      #&text      used to mark proper names and place names
      #             Example: &Fiat           &Joe's &Grill
      #text --    end of interrupted turn and continuation
      #-- text    of same turn after interruption, e.g.
      #             A: I saw &Joe yesterday coming out of --
      #             B: You saw &Joe?!
      #             A: -- the music store on &Seventeenth and &Chestnut. 
      #//text//   aside (talker addressing someone in background)
      #             Example: //quit it, I'm talking to your sister!//
     
      $a=~s/^&(.*)&/$1/g;
      $a=~s/^&(.*)/$1/g;
      $a=~s/^;(.*);/$1/g;
      $a=~s/^;(.*)/$1/g;
      $a=~s/\/\///g;
      $a=~s/--//g;
      $a =~ s/B\~/B/g;
      $a =~ s/B\(t\)/B/g;
      $a =~ s/il\(k\)/il/g;
      $a =~ s/il\(g\)/il/g;
      $a =~ s/il\+\&/il+/g;
      $a =~ s/[+,.?!]$//g;

      print "$a " if $a;
      next;
    }
    if ( ($a =~ /--[^\s]+/ ) || ($a =~ /[^\s]+--/  ) ) {
      print "<UNK> ";
      next;
    }
      
    #This is to get the full "English phrase" into the $a variable
    #In cases where spaces are used instead of underscores...
    if (  ( $a =~ /\<(English|French|MSA|Italian|Delta|Upper|\?)/i ) && ( $a !~ /\<(English|French|MSA|Italian|Delta|Upper|\?)_.*\>/i) ) {
      while (($a !~ /\<(English|French|MSA|Italian|Delta|Upper|\?)_.*\>/i) && ($n < @A )) {
        $a = $a . "_" . $A[$n]; 
        $n += 1;
      }
      if ( $a !~ /\<(English|French|MSA|Italian|Delta|Upper|\?)_.*\>/i ) {
        print STDERR "Could not parse line ${line}Reparsed: $a\n";
        next LINE;
      }
    }

    if ( $a =~ /\<English_.*\>/i ) {
      
      #For Arabic, the English word is often preceded by al-/il-/other 
      $tmp=$a;
      $tmp =~ s/(.*?)\<English_.*\>/$1/i;
      print "$tmp " if $tmp;
      
      $a=~ s/.*?\<English_(.*)\>/$1/i;
      #print "<ENGLISH$a>";
      @words=split("_",uc($a));
      $i=0;
      while ($i < @words) {
        $word=$words[$i];
        $i+=1;
        
        while ($word =~ /(.*)[+.,?!#]/) {
           $word =~ s/(.*)[+.,?!#]/$1/;
        }
        if ($word =~ /\(\(\)\)/ ) {
          print "<UNK> ";
        }
        if ($word =~ /\(\(.*/ ) {
          while (( $word !~ /.*\)\)/ )) {
            #print "<UNK> ";
            $word =~ s/\(\(//g;
            #print "$word ";
            $word =~ /^.+-$/ ? print "<UNK> " : print "$word ";
            $word=$words[$i];
            $i+=1;
          }
          #print "<UNK> ";
          $word =~ s/^\(\(//g;
          $word =~ s/\)\)$//g;
          
          if ( $word ) {
            $word =~ /^.+-$/ ? print "<UNK> " : print "$word ";
          }
          #print "$word " if $word;
          next; # if $i < @words;

        }
        
        if (( $word =~/^-.*$/) || ( $word =~/^.*-$/) ) {
          #text-   partial word
          #        example: absolu-
          #**text**    idiosyncratic word, not in common use, not necessarily 
          #            included in lexicon
          #            Example: **poodle-ish**

          #print STDERR "$word\n";
          print "<UNK> ";
          next;
        }

        if ($word =~ /\[.*/ ) {
          while (( $word !~ /.*\]/ )) {
            $word=$words[$i];
            $i+=1;
          }
          print "<NOISE> ";
          next; # if $i < @words;
        }


        while ($word =~ /(.*)[+.,?!]$/) {
           $word =~ s/(.*)[+.,?!]$/$1/;
        }
        $word=~s/^\&//g;
        print "$word " if $word =~ /[a-zA-Z0-9][-a-zA-Z0-9']*/;
        
      }
      next;
    }

    if ( $a =~ /\<\?.*\>/ ) {
      #Unknown language, no transcription
      print "<UNK> ";
      next;

    }
    if ( $a =~ /\<.*\>/ ) {
      #some language... Might be hard to get correct phonetic transcription
      #so let's just generate UNKs for the time being...
      $a=~s/\<[A-Za-z][^_]*_(.*)>/$1/;

      @words=split("_",uc($a));
      foreach $word(@words) {
        print "<UNK> ";
      }
      next;

    }


    if ( $a =~/\(\(/ ) {
      #((text))            unintelligible; text is best guess at transcription
      #                    ((coffee klatch))
      #(( ))               unintelligible; can't even guess text
      #                    (( ))
      $tmp=$a;
      print "<UNK> ";
      while (( $a !~ /.*\)\)/ ) && ($n < @A)) {
        #print Dumper(\@A, $tmp, $a, $n);
        $a = $A[$n]; 
        $n += 1;
        $tmp = "$tmp $a";
        print "<UNK> ";
      }
      $unint{$tmp} += 1;
      next;
    }
    #if ($word =~ /\[.*/ ) {
    #  while (( $word !~ /.*\]/ )) {
    #    $word=$words[$i];
    #    $i+=1;
    #  }
    #  print "<NOISE> ";
    #  next; # if $i < @words;
    #}
    
    $a =~ s/B\~/B/g;
    $a =~ s/B\(t\)/B/g;
    $a =~ s/il\(k\)/il/g;
    $a =~ s/il\(g\)/il/g;
    $a =~ s/(.+)\+&/$1+/g;
    $a =~ s/\+(.+)$/$1/g;
    $a =~ s/^;(.+)$/$1/g;
    $a =~ s/[+,.?!]$//g;

    #print STDERR "X: $a\n" unless $a =~ /^[+,.?!#]$/;
    print "$a " unless $a =~ /^[+,.?!#]$/;
  }
  print "\n";
}

#print Dumper(\%tags);
#print Dumper(\%langs);
#print Dumper(\%noises);
#print Dumper(\%unint);
