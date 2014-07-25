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

while (<STDIN>) {
  $line=$_;
  @A = split(" ", $line);
  print "$A[0] ";

  $n = 1;
  while ($n < @A ) {
    $a = $A[$n]; 
    $n+=1;

    if ( grep { $_ eq $a } @vocalized_noise ) {
      print "<V-NOISE> ";
      next;
    }

    if (($a eq "{laugh}")) {
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
    if (( $a =~/^-[^-]*$/) || ( $a =~/^[^-]+-$/) || ($a =~ /\*.*\*$/ ) ) {
      #text-   partial word
      #        example: absolu-
      #**text**    idiosyncratic word, not in common use, not necessarily 
      #            included in lexicon
      #            Example: **poodle-ish**

      print "<UNK> ";
      next;
    }

    if (( $a =~ /\+.*\+/) || ( $a =~/\@.*\@/)) {
      #+text+  mandarinized foreign name, no standard spelling
      #@text@  a abbreviation word
      
      $a =~ s/[+@](.*)[+@]/$1/;
      print "$a ";
      next;
    }
    if ( $a =~/^%/ ) {
      #%text         This symbol flags non-lexemes, which are
      #              general hesitation sounds.
      #              Example: %mm %uh
      $noises{$a} += 1;
      print "<HES> ";
      next;
    }

    if (( $a =~/^\&/ ) || ($a eq "--") || ($a eq "//" ) ) {
      #&text      used to mark proper names and place names
      #             Example: &Fiat           &Joe's &Grill
      #text --    end of interrupted turn and continuation
      #-- text    of same turn after interruption, e.g.
      #             A: I saw &Joe yesterday coming out of --
      #             B: You saw &Joe?!
      #             A: -- the music store on &Seventeenth and &Chestnut. 
      #//text//   aside (talker addressing someone in background)
      #             Example: //quit it, I'm talking to your sister!//
     
      $a=~s/^&(.*)&/$1/;
      $a=~s/^&(.*)/$1/;
      $a=~s/\/\///;
      $a=~s/--//;
        
      
      print "$a " if $a;
      next;
    }
    
    if ( $a =~ /\<English_.*\>/ ) {
      $a=~ s/\<English_(.*)\>/$1/;
      @words=split("_",uc($a));
      $i=0;
      while ($i < @words) {
        $word=$words[$i];
        $i+=1;
        
        while ($word =~ /(.*)[+.,?!#]/) {
           $word =~ s/(.*)[+.,?!#]/$1/;
        }
        if ($word =~ /\(\(.*/ ) {
          while (( $word !~ /.*\)\)/ )) {
            print "<UNK> ";
            $word=$words[$i];
            $i+=1;
          }
          print "<UNK> ";
          next; # if $i < @words;

        }
        
        if (( $word =~/^-.*$/) || ( $word =~/^.*-$/) ) {
          #text-   partial word
          #        example: absolu-
          #**text**    idiosyncratic word, not in common use, not necessarily 
          #            included in lexicon
          #            Example: **poodle-ish**

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


        while ($word =~ /(.*)[+.,?!]/) {
           $word =~ s/(.*)[+.,?!]/$1/;
        }

        print "$word " if $word =~ /[a-zA-Z0-9][-a-zA-Z0-9']*/;
        
      }
      next;
    }

    if ( $a =~ /\<?.*\>/ ) {
      #Unknown language, no transcription
      print "<UNK> ";

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

    print "$a " unless $a =~ /[+,.?!#]/;
  }
  print "\n";
}

#print Dumper(\%tags);
#print Dumper(\%langs);
#print Dumper(\%noises);
#print Dumper(\%unint);
