s/.*<cross_talk_begin>.*/ /g
s/.*<cross_talk_end>.*/ /g
s/.*<noise_begin>.*/ /g
s/.*<noise_end>.*/ /g
s/.*<cross_talk>.*/ /g
s/.*<background_noise_begin>.*/ /g
s/.*<background_noise_end>.*/ /g
s/.*<emotion_begin>.*/ /g
s/.*<emotion_end>.*/ /g
s/.*<speech_cutoff>.*/ /g
s/.*<stumble>.*/ /g
s/.*<distortion>.*/ /g
s/.*<laugh_begin>.*/ /g
s/.*<laugh_end>.*/ /g
s/.*<end>.*/ /g
s/.*<distortion_begin>.*/ /g
s/.*<distortion_end>.*/ /g
s/.*<preliminary_talk_end>.*/ /g
s/.*<solo violin music from 3\:55 minute to 5\:00 minute>.*/ /g
s/.*<sound_problem_begin>.*/ /g
s/.*<sound_problem_end>.*/ /g
s/.*<static_end>.*/ /g
s/.*<stuttering>.*/ /g
s/.*<aircraft_noise_begin>.*/ /g
s/.*<singing>.*/ /g
s/<distortion>//g
s/<whisper>//g
s/<static>//g
s/<s>//gi
