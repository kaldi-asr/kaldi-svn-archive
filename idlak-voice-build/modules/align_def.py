#!/usr/bin/python

# Copyright 2013 CereProc Ltd.

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

# Triphone speaker specific aligner using kaldi

import sys, os.path, time, subprocess, glob
from xml.dom.minidom import getDOMImplementation

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Triphone speake specific aligner using kaldi'
FRAMESHIFT = 0.005
NOSTATES = 6
STARTENDMINSIL = 0.01
# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

def get_wav_durations(kaldidir, wavscp):
    wavs = open(wavscp).readlines()
    durations = {}
    for w in wavs:
        w = w.split()[1]
        stem = os.path.splitext(os.path.split(w)[1])[0]
        pipe = subprocess.Popen([os.path.join(kaldidir, 'src', 'featbin', "wav-info"), '--print-args=false', w], stdout=subprocess.PIPE)
        duration = pipe.communicate()[0].split('\n')[1].split()[1]
        durations[stem] = duration
    return durations
        
def gettimes(frames, framesz = 0.01):
    times = []
    t = 0.0
    frames = frames.replace(' ]', '')
    frames = frames.split(' [')
    for p in frames[1:]:
        pp = p.strip().split()
        dur = len(pp) * framesz
        times.append("%.3f %.3f" % (t, t + dur))
        t  += dur
    return times

def write_as_labs(kaldialign, frameshift, wavdurations, dirout):
    lines = open(kaldialign).readlines()
    for l in lines:
        ll = l.split()
        uttid = ll[0]
        duration = float(wavdurations[uttid])
        fp = open(dirout + '/' + uttid + '.lab', 'w')
        labs = ll[1:]
        prvphon = None
        start = 0.0
        time = 0.0
        for p in labs:
            if p == 'sp': p = 'sil_S'
            if prvphon and p != prvphon:
                fp.write('%.3f %.3f %s\n' % (start, time, prvphon.split('_')[0]))
                start = time
            prvphon = p
            time += frameshift
        fp.write('%.3f %.3f %s\n' % (start, time, prvphon.split('_')[0]))
        fp.close()            
            
def write_as_wrdlabs(kaldialign, wavdurations, labdir, dirout):
    lastid = ''
    # read data into lists by file id
    lines = open(kaldialign).readlines()
    wrds = {}
    wrdtimes = []
    for l in lines:
        ll = l.split()
        if ll[0] != lastid:
            if lastid:
                wrds[lastid] = wrdtimes
                wrdtimes = []
            lastid = ll[0]
        wrdtimes.append([float(ll[2]), float(ll[3]), ll[4].lower()])
    wrds[lastid] = wrdtimes
    # write out adding silences were required
    fileids = wrds.keys()
    fileids.sort()
    for fileid in fileids:
        lasttime = 0.0
        fp = open(dirout + '/' + fileid + '.wrd', 'w')
        #print fileid
        #print wrds[fileid]
        for wrd in wrds[fileid]:
            if wrd[2] != '<sil>':
                if wrd[0] > lasttime and ("%.3f" % wrd[0]) != ("%.3f" % lasttime):
                    fp.write("%.3f %.3f SIL\n" % (lasttime, wrd[0]))
                fp.write("%.3f %.3f %s\n" % (wrd[0], wrd[0] + wrd[1], wrd[2]))
                lasttime = wrd[0] + wrd[1]
        labs = open(os.path.join(labdir, fileid + '.lab')).readlines()
        etime = float(labs[-1].split()[1])
        if etime > lasttime and  ("%.3f" % etime) != ("%.3f" % lasttime):
            fp.write("%.3f %.3f SIL\n" % (lasttime, etime))
        fp.close()
        
def write_as_statelabs(kaldialign, frameshift, nstates, wavdurations, labdir, dirout):
    lines = open(kaldialign).readlines()
    for l in lines:
        ll = l.split()
        uttid = ll[0]
        plabs = open(os.path.join(labdir, uttid + '.lab')).readlines()
        fp = open(dirout + '/' + uttid + '.stt', 'w')
        labs = ll[1:]
        prvstt = None
        start = 0.0
        time = 0.0
        p = 0
        for stt in labs:
            if prvstt and stt != prvstt:
                fp.write('%.3f %.3f %s\n' % (start, time, prvstt))
                start = time
            prvstt = stt
            time += frameshift
        fp.write('%.3f %.3f %s\n' % (start, time, prvstt))
        fp.close()           
    
# TODO given the silence duration and location it is possible to look at f0
# across recent phonetic material to decide if a break is a rising or falling
def write_xml_textalign(breaktype, breakdef, labdir, wrddir, output):
    impl = getDOMImplementation()

    document = impl.createDocument(None, "document", None)
    doc_element = document.documentElement
    
    labs = glob.glob(labdir + '/*.lab')
    labs.sort()
    f = open(output, 'w')
    f.write('<document>\n')
    for l in labs:
        stem = os.path.splitext(os.path.split(l)[1])[0]

        fileid_element = document.createElement("fileid")
        doc_element.appendChild(fileid_element)
        fileid_element.setAttribute('id', stem)
        
        words = open(os.path.join(wrddir, stem + '.wrd')).readlines()
        phones = open(l).readlines()
        pidx = 0
        for widx, w in enumerate(words):
            ww = w.split()
            pron = []
            while 1:
                pp = phones[pidx].split()
                if pp[1] != ww[1] and float(pp[1]) > float(ww[1]):
                    break
                pron.append(pp[2])
                pidx += 1
                if pidx >= len(phones):
                    break
            if ww[2] != 'SIL':
                lex_element = document.createElement("lex")
                fileid_element.appendChild(lex_element)
                lex_element.setAttribute('pron', ' '.join(pron))
                
                text_node = document.createTextNode(ww[2])
                lex_element.appendChild(text_node)
            else:
                if not widx or (widx == len(words) - 1):
                    break_element = document.createElement("break")
                    fileid_element.appendChild(break_element)
                    break_element.setAttribute('type', breakdef)
                else:
                    btype = breakdef
                    for b in breaktype.split(','):
                        bb = b.split(':')
                        minval = float(bb[1])
                        if float(ww[1]) - float(ww[0]) < minval:
                            btype = bb[0]
                    break_element = document.createElement("break")
                    fileid_element.appendChild(break_element)
                    break_element.setAttribute('type', btype)
        f.write(fileid_element.toxml() + '\n')

    f.write('</document>')
    f.close()
                
def main():
    # process the options based on the default build configuration
    build_conf, parser = build_configuration.get_config(SCRIPT_NAME, DESCRIPTION, SCRIPT_NAME)
    #print 'SEQ', build_conf.dataseq
    # parse commamd line
    if __name__ == '__main__':
        opts, args = parser.parse_args()
        # and load custom configurations
        if opts.bldconf:
            build_conf.parse(opts.bldconf)
        if opts.spkconf:
            build_conf.parse(opts.spkconf)
        else:
            parser.error("Speaker configuration is required e.g. speaker_conf/bdl.xml")
            
        build_conf.updatefromopts(opts)
    # set up logging, check idlak-scratch, check dependencies and build as required
    build_conf.set_build_environment(SCRIPT_NAME)

    if opts.flist:
            build_conf.logger.log('warn', 'flist does NOT currently work in align_def.py')

    # ADD MODULE SPECIFIC CODE HERE
    # get required input files from idlak-data
    spkdir = os.path.join(build_conf.idlakdata, build_conf.lang, build_conf.acc, build_conf.spk)
    # get required directories from dependent modules
    alignsetupdir = build_conf.get_input_dir('alignsetup_def')
    # examine module specific settings and set as appropriate
    breaktype = build_conf.getval('align_def', 'break')
    breakdef = build_conf.getval('align_def', 'breakdef')
    # process dat
    # remove old setup data 
    com = 'rm -rf %s' % (os.path.join(build_conf.outdir, 'output', 'data'))
    build_conf.logger.log('info', 'Removing old alignsetup information: %s' % (com))
    os.system(com)
    # copy setup data
    com = 'cp -R %s %s' % (alignsetupdir, os.path.join(build_conf.outdir, 'output', 'data'))
    build_conf.logger.log('info', 'Copying alignsetup information: %s' % (com))
    os.system(com)
    # link conf, steps and utils directories from egs/wsj/s5
    com = 'ln -s %s %s' % (os.path.join(build_conf.kaldidir, 'egs', 'wsj', 's5', 'conf'),
                           os.path.join(build_conf.outdir, 'output', 'conf'))
    build_conf.logger.log('info', 'Linking wsj s5 conf: %s' % (com))
    os.system(com)
    com = 'ln -s %s %s' % (os.path.join(build_conf.kaldidir, 'egs', 'wsj', 's5', 'utils'),
                           os.path.join(build_conf.outdir, 'output', 'utils'))
    build_conf.logger.log('info', 'Linking wsj s5 utils: %s' % (com))
    os.system(com)
    com = 'ln -s %s %s' % (os.path.join(build_conf.kaldidir, 'egs', 'wsj', 's5', 'steps'),
                           os.path.join(build_conf.outdir, 'output', 'steps'))
    build_conf.logger.log('info', 'Linking wsj s5 steps: %s' % (com))
    os.system(com)
    # update path for kaldi scripts
    pathlist = [os.path.join(build_conf.outdir, 'output', 'utils'),
                os.path.join(build_conf.kaldidir, 'src', 'featbin'),
                os.path.join(build_conf.kaldidir, 'src', 'bin'),
                os.path.join(build_conf.kaldidir, 'src', 'fstbin'),
                os.path.join(build_conf.kaldidir, 'tools', 'openfst', 'bin'),
                os.path.join(build_conf.kaldidir, 'src', 'latbin'),
                os.path.join(build_conf.kaldidir, 'src', 'lm'),
                os.path.join(build_conf.kaldidir, 'src', 'sgmmbin'),
                os.path.join(build_conf.kaldidir, 'src', 'sgmm2bin'),
                os.path.join(build_conf.kaldidir, 'src', 'fgmmbin'),
                os.path.join(build_conf.kaldidir, 'src', 'nnetbin'),
                os.path.join(build_conf.kaldidir, 'src', 'nnet-cpubin'),
                os.path.join(build_conf.kaldidir, 'src', 'kwsbin'),
                os.path.join(build_conf.kaldidir, 'src', 'gmmbin')]
    os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
    datadir = os.path.join(build_conf.outdir, 'output', 'data')
    # create lang directory using kaldi script
    com = "cd %s/output; utils/prepare_lang.sh --num-nonsil-states %d data '<OOV>' data/lang data/lang" % (build_conf.outdir, NOSTATES)
    build_conf.logger.log('info', 'running kaldi script to build lang subdir')
    os.system(com)
    # extract mfccs
    #com = "cd %s/output; steps/make_mfcc.sh --nj 1 data/train data/mfcc_log data/mfcc" % (build_conf.outdir)
    #build_conf.logger.log('info', 'running kaldi script to extract mfccs')
    build_conf.logger.log('info', 'making mfcc directory')
    mfccdir = os.path.join(build_conf.outdir, 'output', 'data', 'mfcc')
    if not os.path.isdir(mfccdir):
        os.mkdir(mfccdir)
    build_conf.logger.log('info', 'extracting mfccs')
    com = "cd %s/output; compute-mfcc-feats --frame-shift=%d --verbose=0 --config=%s scp:%s ark:- | copy-feats --compress=false ark:- ark,scp:%s,%s" % (build_conf.outdir, int(FRAMESHIFT * 1000), "conf/mfcc.conf", "data/train/wav.scp", "data/mfcc/raw_mfcc_train.1.ark", "data/mfcc/raw_mfcc_train.1.scp")
    os.system(com)
    # build dummy spk to utt file
    com = "cd %s/output; utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt" % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute dummy spk2utt file')
    os.system(com)
    # compute feature stats
    #copy scp file to train/feats.scp
    build_conf.logger.log('info', 'copying mfcc scp to feats scp')
    com = "cd %s/output; cp data/mfcc/raw_mfcc_train.1.scp data/train/feats.scp" % (build_conf.outdir)
    os.system(com)
    com = "cd %s/output; steps/compute_cmvn_stats.sh data/train data/mfcc data/mfcc" % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute feature statistics')
    os.system(com)
    # mono train
    com = 'cd %s/output; steps/train_mono.sh --nj 1 data/train data/lang kaldimono_output' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute flat start monophone models')
    os.system(com)
    # delta train (triphone)
    com = 'cd %s/output; steps/train_deltas.sh 2000 10000 3 data/train data/lang kaldimono_output kaldidelta_tri_output' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute flat start triphone models')
    os.system(com)
    # delta train (quinphone)
    com = 'cd %s/output; steps/train_deltas.sh 2000 10000 5 data/train data/lang kaldidelta_tri_output kaldidelta_quin_output' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute flat start quinphone models')
    os.system(com)
    # extract the phone alignment
    com = 'cd %s/output; ali-to-phones --per-frame kaldidelta_quin_output/35.mdl "ark:gunzip -c kaldidelta_quin_output/ali.1.gz|" ark,t:- |  utils/int2sym.pl -f 2- data/lang/phones.txt > align.dat' % (build_conf.outdir)
    # com = 'cd %s/output; show-alignments data/lang/phones.txt kaldidelta_quin_output/35.mdl "ark:gunzip -c kaldidelta_quin_output/ali.1.gz|" > align.dat' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to extract alignment')
    os.system(com)
    # extract the state alignment
    com = 'cd %s/output; ali-to-hmmstate kaldidelta_quin_output/35.mdl "ark:gunzip -c kaldidelta_quin_output/ali.1.gz|" ark,t:- > sttalign.dat' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to extract state alignment')
    os.system(com)
    #extract the word alignment
    com = 'cd %s/output; linear-to-nbest "ark:gunzip -c kaldidelta_quin_output/ali.1.gz|" "ark:utils/sym2int.pl --map-oov 1669 -f 2- data/lang/words.txt < data/train/text |" \'\' \'\' ark:- | lattice-align-words data/lang/phones/word_boundary.int kaldidelta_quin_output/35.mdl ark:- ark:- | nbest-to-ctm --frame-shift=%f --precision=3 ark:- - | utils/int2sym.pl -f 5 data/lang/words.txt > wrdalign.dat' % (build_conf.outdir, FRAMESHIFT)
    build_conf.logger.log('info', 'running kaldi scripts to extract word alignment')
    os.system(com)
    # get actual duration times of all wav files
    build_conf.logger.log('info', 'Collecting wav file durations')
    wavdurations = get_wav_durations(build_conf.kaldidir, os.path.join(build_conf.outdir, 'output', 'data', 'train', 'wav.scp'))
    # write alignment as files that are readbale by wavesurfer etc for checking
    build_conf.logger.log('info', 'Writing lab and wrd files')
    labdir = os.path.join(build_conf.outdir, 'output', 'labs')
    if not os.path.isdir(labdir):
        os.mkdir(labdir)
    write_as_labs(os.path.join(build_conf.outdir, 'output', 'align.dat'), FRAMESHIFT, wavdurations, labdir)
    wrddir = os.path.join(build_conf.outdir, 'output', 'wrds')
    if not os.path.isdir(wrddir):
        os.mkdir(wrddir)
    write_as_wrdlabs(os.path.join(build_conf.outdir, 'output', 'wrdalign.dat'), wavdurations, labdir, wrddir)
    statedir = os.path.join(build_conf.outdir, 'output', 'stts')
    if not os.path.isdir(statedir):
        os.mkdir(statedir)
    write_as_statelabs(os.path.join(build_conf.outdir, 'output', 'sttalign.dat'), FRAMESHIFT, NOSTATES, wavdurations, labdir, statedir)
    #write alignment based xml text file
    write_xml_textalign(breaktype, breakdef, labdir, wrddir, os.path.join(build_conf.outdir, 'output', 'text.xml'))
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
