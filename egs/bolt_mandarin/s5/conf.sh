extra_decoding_opts=(--num-threads 4 --parallel-opts '-pe smp 4' )
train_nj=32
decode_nj=12

# The path to training corpora:
CALLHOME_MA_CORPUS_A=/export/corpora/LDC/LDC96S34/CALLHOME/
CALLHOME_MA_CORPUS_T=/export/corpora/LDC/LDC96T16/

HKUST_MA_CORPUS_A=/export/corpora/LDC/LDC2005S15/
HKUST_MA_CORPUS_T=/export/corpora/LDC/LDC2005T32

HUB5_MA_CORPUS_A=/export/corpora/LDC/LDC98S69
HUB5_MA_CORPUS_T=/export/corpora/LDC/LDC98T26

RT04F_MA_TRAIN_CORPUS_A=/export/corpora/LDC/LDC2004E69/train04f
RT04F_MA_TRAIN_CORPUS_T=/export/corpora/LDC/LDC2004E70/reference/train04f/
RT04F_MA_DEV_CORPUS_A=/export/corpora/LDC/LDC2004E67/package/rt04_stt_man_dev04f/
RT04F_MA_DEV_CORPUS_T=/export/corpora/LDC/LDC2004E68/LDC2004E68_V1_0/reference/dev04f/


