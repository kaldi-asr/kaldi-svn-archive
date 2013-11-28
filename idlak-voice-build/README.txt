IDLAK VOICE BUILDING

Objectives:

Do not have to rebuild everything if hit an error

Modular

Audit trail (logging)

ability to configure and modify by voice and by architecture and by
command line

Easy to add modules and alternative build sequences

One button voice build

Not directory specific

Clear what data belongs to what

 /afs/inf.ed.ac.uk/user/m/matthewa/kaldi/matthewa/kaldi-idlak/src/bin/myconvert-fullctx kaldidelta_quin_output/final.mdl "ark:gunzip -c kaldidelta_quin_output/ali.1.gz|" ark,t:../../../cex_def/20131024/output/cex.ark ark,t:temp2


Still not matching phone lengths (silence problem)

Hacked cex to remove double sils and kill all features to 0.

wrote script to transform hts style qset into a intermidiate XML
version which is useable by cex_def.py indexed by teh context
extraction function name. WARNING this will only include questions
which have a matching HTS name/function in the hts archtecture used to
build it. In general we'd imagine this was hand coded.

cex adds phone contexts as extra features so that qset lookups will
work for context (so final data will have it twice once from kaldi
once from idlak)

cex now generates kaldi format q set which matches context extraction.

Noted that cmu lex doesn't have ax and uses hh instead of h (hacked in
kaldi qset generation)

Alignment seems okay, merge with cex data works:

need to check in cex_def, document align_def and cex_def as they now
stand and check utils for doing tree accumulation before checking in.

STILL need to update config system and module .h




