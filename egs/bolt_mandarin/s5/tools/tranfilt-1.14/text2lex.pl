#!/usr/local/bin/perl
# written by Ken Ross knross@bbn.com
# 9/15/95
# This script searches the LDC dictionary for the lemma of a particular 
#  word (or list of words) and prints out the lemma (or list of lemmas) for 
#  each word on a separate line.  If the word is not in the dictionary, 
#  the word itself gets printed out.  
# 9/25/95
# Adapted by Jon Fiscus for the LVCSR evaluation:
# Version 1.1 - change it to pass comment lines through un-harmed


$Version="1.1";

$Usage="Usage: text2lex -d LDC_dictionary -i ctm|stm infile|- outfile|-\n".
"Version: $Version\n".
"\n".
"Desc:  text2lex uses the LDC Spanish dictionary to convert the lexical\n".
"       items in the input file to lexemes.  The output is suitable for\n".
"       scoring lexeme accuracy via the sclite scoring program\n".
"\n".
"       OPTIONS :==\n".
"       -d Dict      Identifies the filename of the Spanish LDC Dictionary\n".
"       -i ctm|stm   Identify the input file format.\n".
"\n";



# Get the command line arguments
require "getopts.pl";
&Getopts('d:i:');
if (!defined($opt_i)) { &errexit("Input File type required"); }
if (!defined($opt_d)) { &errexit("LDC Spanish Dictionary required"); }

##the following variable points to the location of the LDC dictionary
$LDC_DIC = $opt_d;
$InputType = $opt_i;

if ($InputType !~ /^ctm$/ && $InputType !~ /^stm$/){
    &errexit("Illegal input file format");
}

if ($#ARGV > 1) { &errexit("Too many arguements"); }
if ($#ARGV == 0) { &errexit("Output Not Specified"); }
if ($#ARGV == -1) { &errexit("Input and Output Not Specified"); }

$InFile=$ARGV[0];
$OutFile=$ARGV[1];

&load_LDC_Dictionary($LDC_DIC);
open(IN,"<$InFile") ||& errexit("Can't open input file $InFile");
open(OUT,">$OutFile") ||& errexit("Can't open input file $OutFile");

while (<IN>){
    if ($_ =~ /^;;/) {
	# Pass comments through without modification
	print OUT;
    } elsif ($InputType =~ /ctm/) {
	chop;
	($f,$c,$bt,$dt,$text,$conf) = split;
	if ($conf !~ /^$/){ 
	    $conf =~ s/^/ /;
	}
	$text =~ s/^ +//g;
	$lexemes = &lexeme_lookup($text);
    	$lexemes =~ tr/a-z/A-Z/;
    	$lexemes =~ tr/\340-\377/\300-\337/;     #Upper-Case accented letters
	$lexemes =~ s/^ +//g;
	$lexemes =~ s/ +$//g;
	if ($lexemes =~ / /){
	    print OUT "$f $c * * <ALT_BEGIN>\n";
	    local (@arr) = split(/\s+/, $lexemes);
	    for ($i=0; $i<=$#arr; $i++){
		print OUT "$f $c $bt $dt $arr[$i]$conf\n";
		if ($i != $#arr){
		    print OUT "$f $c * * <ALT>\n";
		}
	    }
	    print OUT "$f $c * * <ALT_END>\n";
	} else {
	    print OUT "$f $c $bt $dt $lexemes$conf\n";
	}
    } elsif ($InputType =~ /stm/) {
	local($new) = "";
	local($start);
	chop;
	s/([{}\/])/ \1 /g;	# make sure all alternates have whitespace
	local (@words) = split;
	$new = $words[0]." ".$words[1]." ".$words[2];
	$start = 3;
	if ($words[$start] =~ /^[0-9.]*/){ 
	    $new = $new." ".$words[$start];
	    $start ++;
	    if ($words[$start] =~ /^[0-9.]*/){ 
		$new = $new." ".$words[$start];
		$start ++;
	    }
	}
	for ($i=$start; $i<=$#words; $i++){
	    $lexemes = &lexeme_lookup($words[$i]);
	    $lexemes =~ tr/a-z/A-Z/;
	    $lexemes =~ tr/\340-\377/\300-\337/; #Upper-Case accented letters
	    $lexemes =~ s/^ +//g;
	    $lexemes =~ s/ +$//g;
	    if ($lexemes =~ / /){
		$lexemes =~ s/ +/ \/ /g;
		$lexemes =~ s/^/{ /g;
		$lexemes =~ s/$/ }/g;
	    }	    
	    $new = $new." ".$lexemes;
	}
	print OUT "$new\n";
    }
}
close (IN);
close (OUT);
exit;

sub lexeme_lookup{
    local($out) = "";
    local($i, $j, $flag, @stem_list, $on_list);
    local($word) = $_[0];
    $word =~ tr/A-Z/a-z/;
    $word =~ tr/\300-\337/\340-\377/;     #Lower-Case accented letters
    local($morph) = $Wordset{$word};

    undef @stem_list;
    $flag = 0;			#set flag to not found
    if ($morph !~ /^$/){
	# print $word, " ", $morph, "\n";
	@morphs = split('//',$morph); # multiple morphological 
	## derivatives separted by double slashes 
	for($i=0; $i <= $#morphs; $i++)
	{
	    @stem = split(/\+/, $morphs[$i]); # fields separated by plus
	    $on_list = 0;
	    $stem[0] =~ tr/A-Z/a-z/;
	    
	    for($j=0; $j <= $#stem_list; $j++)
	    {
		if($stem_list[$j] eq $stem[0]) # stem in 1st field
		{
		    $on_list = 1; # want to print only once
		}
	    }
	    if( $on_list == 0 ) # not on list yet
	    {
		push(@stem_list, $stem[0]);
	    }
	}
	$flag = 1; #found flag set
    }
    if($flag == 1)
    {
	for($i=0; $i <= $#stem_list; $i++)
	{
	    $out = $out." ".$stem_list[$i];
	}
	undef @stem_list;	# clear out for the next time through
	$flag = 0;
    }
    else
    {
	# never found the word
	$out = $word;
    }				
    $out;
}



####
####  Read in the LDC Spanish dictionary
####  Inputs:   - Argument 0: Filename of the LDC Dictionary
####  Outputs:  - Return nothing
####            - Create a global associative array called $Wordset.
####              It contains the morphology of each word.
sub load_LDC_Dictionary{
    local($DIC) = $_[0];
    local ($word, $morph, $pronoun, $stress, $callhome, $radio, $ap_news, 
	   $reuters, $norte);

    open(TMPBUF,$DIC) || die "Can't open LDC dictionary $DIC\n";
    while (<TMPBUF>) 
    {
	($word, $morph, $pronoun, $stress, $callhome, $radio, $ap_news, 
	 $reuters, $norte)= split(/\s+/,$_);
	#a very unseemly hack to deal with the fact
	#the dictionary is case sensitive while the recognizer
	#output isn't.
	$word =~ tr/A-Z/a-z/;
	$word =~ tr/\300-\337/\340-\377/;     #Lower-Case accented letters
	$Wordset{$word} = $morph;
    }
    close TMPBUF;
}

sub errexit{
    local($mesg) = $_[0];
    print $Usage;
    print "$mesg\n\n";
    exit;
}
	    
