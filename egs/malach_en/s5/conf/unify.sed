s/<bkgrd/<background/gi
s/<bkgd/<background/gi
s/<bkdg/<background/gi
s/<bkrd/<background/gi
s/<bkdg/<background/gi
s/<bgrd/<background/gi
s/<bkrgd/<background/gi
s/<bkrnd/<background/gi
s/<bacground/<background/gi
s/<backround/<background/gi
s/<background[ -][ -]*noise>/<background_noise>/gi
s/ends>/end>/gi
s/start>/begin>/gi
s/starts>/begin>/gi
s/begins>/begin>/gi

s/<sniff>/<emotion>/gi
s/<sob>/<emotion>/gi

s/<whispers*>/<whisper>/gi

s/background noise ends*/background_noise_end/gi
s/background noise begins*/background_noise_begin/gi
s/<background[-_ ]noise[-_ ]and[-_ ]talk>/<background_noise>/gi
s/<NOISE!*>/<noise>/gi
s/<chirp[a-z]*>/<noise>/gi
s/<background_noise>/<noise>/gi
s/<noise_on>/<noise_begin>/gi
s/<noise_off>/<noise_end>/gi
s/<static_noise>/<noise>/gi
s/<exhale_noise>/<breath>/gi
s/<exhale>/<breath>/gi
s/<puf>/<breath>/gi

s/<noise of children playing>/<noise>/gi
s/<noises>/<noise>/gi
s/<rattling_noise_begin>/<background_noise_begin>/gi
s/<street_noise>/<noise>/gi
s/<mic_noise>/<noise>/gi
s/<mic_noise_begin>/<noise_begin>/gi
s/<mic_noise_end>/<noise_end>/gi
s/<buzz_noise_begin>/<noise_begin>/gi
s/<buzz_noise_end>/<noise_end>/gi
s/<phone>/<noise>/g
s/<drink>/<noise>/g
s/<singing>/<singing>/gi
s/<sing>/<singing>/gi
s/<sings>/<singing>/gi
s/<nosie>/<noise>/g
s/<sniffle>/<noise>/g

s/<Pause>/<pause>/gi
s/<long_pause>/<pause>/gi
s/<very_long_pause>/<pause>/gi
s/<speech_pause>/<pause>/gi
s/<slight_pause>/<pause>/gi
s/<empty>/<pause>/gi

s/<cross_cough_begin>/<background_cough_begin>/gi
s/<cross_cough_end>/<background_cough_end>/gi
s/<couth>/<cough>/gi

s/<snorts>/<breath>/gi
s/<Breath>/<breath>/gi
s/<i*nhales*>/<breath>/gi
s/<background_inhale>/<breath>/gi
s/<background_breath_begin>/<breath>/gi
s/<background_breath_end>/<breath>/gi
s/<breath sucked in>/<breath>/gi
s/<breath_begin>/<breath>/gi
s/<breath_end>/<breath>/gi
s/<sigh>/<breath>/gi

s/<silenc>/<silence>/gi
s/<silence of sorts>/<silence>/gi

s/<background_speech>/<cross_talk>/gi
s/<cross_hm>/<cross_talk>/gi
s/<cross_hum>/<cross_talk>/gi
s/<cross_huh>/<cross_talk>/gi
s/<cross_laugh_begin>/<cross_talk>/gi
s/<cross_laugh_end>/<cross_talk>/gi
s/<cross_over_begin>/<cross_talk>/gi
s/<cross_sniff>/<cross_talk>/gi
s/<cross_uh-hm>/<cross_talk>/gi

s/<lip_smack>/<mouth>/gi
s/<sucking_sound>/<mouth>/gi
s/<clicks*>/<mouth>/gi
s/<clear_throat>/<mouth>/gi
s/<swallow>/<mouth>/gi

s/<sound>/<noise>/gi
s/<speech_cutoff_and_chopped_badly>/<speech_cutoff>/gi
s/<speech_cutout>/<speech_cutoff>/gi

s/<heavy breath[ing]*>/<breath>/gi
s/<break>/<breath>/gi

s/<unitelligible/<unintelligible/gi
s/<uniintelligible/<unintelligible/gi
s/<uintelligible/<unintelligible/gi
s/<intelligible>/<unintelligible>/gi
s/\[ <unintelligible> \]/<unintelligible>/gi
s/\[unintelligible\]/<unintelligible>/gi
s/(unintelligible)/<unintelligible>/gi
s/@<unintelligible>/<unintelligible>/gi
s/@<unknown>/<unintelligible>/gi
s/<mumble>/<unintelligible>/gi

s/<cross_speech/<cross_talk/gi
s/<cros_talk/<cross_talk/gi
s/<cross[ -]talk/<cross_talk/gi
s/[- ][- ]*begin>/_begin>/gi
s/[- ][- ]*end>/_end>/gi
s/<long pause>/<pause>/gi
s/<laughs*>/<laugh>/gi
s/<breathe>/<breath>/gi
s/<inahle>/<breath>/gi

s/<clear throat>/<clear_throat>/gi
s/<clears throat>/<clear_throat>/gi


s/\[foreign[ -_]words*[ a-z]*\]/<foreign>/gi 
s/\[foreign[ -_]lang\]/<foreign>/gi 
s/\[foreign[ -_]name\]/<foreign>/gi 
s/\[foreign\]/<foreign>/gi
s/<foreign_[a-z]*>/<foreign>/gi

s/<background_talk>/<background_noise>/gi
s/<background_meow>/<background_noise>/gi
s/<background_laugh>/<background_noise>/gi
s/<background_sniff>/<background_noise>/gi
s/<background>/<background_noise>/gi
s/<background_hum>/<background_noise>/gi
s/<background_bang>/<background_noise>/gi
s/<background_buzz>/<background_noise>/gi
s/<background_hm>/<background_noise>/gi
s/<background_knocking>/<background_noise>/gi
s/<background_ringing>/<background_noise>/gi
s/<background_s->/<background_noise>/gi
s/<background_sniffling>/<background_noise>/gi
s/<background_[a-z]*_begin>/<background_noise_begin>/gi
s/<background_[a-z]*_end>/<background_noise_end>/gi

#Second stage of cleanup
s/<foreign>/<unintelligible>/gi
s/<clear_throat>/<cough>/gi
s/<background_noise>/<noise>/gi
s/<background_cough>/<cough>/gi
s/<background_breath>/<breath>/gi
#Final cleanup
s/[\(\)]/ /gi
#s/\[\|\]/ /gi
s/@ /@/gi
s/   */ /gi
s/^  *//gi
s/  *$//gi


#In addition, let's map <unintelligible> to <UNK>
s/@*<unintelligible>/<UNK>/gi

