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

# Monophone speake specific aligner using kaldi

import sys, os.path, time, subprocess, glob

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Monophone speake specific aligner using kaldi'

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
        pipe = subprocess.Popen([os.path.join(kaldidir, 'src', 'idlakfexbin', "wavinfo"), '--print-args=false', w], stdout=subprocess.PIPE)
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

def write_as_labs(kaldialign, wavdurations, dirout):
    lines = open(kaldialign).readlines()
    for lno in range(0, len(lines), 3):
        times = gettimes(lines[lno])
        labs = lines[lno + 1].split()
        uttid = labs[0]
        labs = labs[1:]
        duration = wavdurations[uttid]
        duration = "%.3f" % float(duration)
        # fix end point
        if labs[-1] == 'sp':
            start, end = times[-1].split()
            end = duration
            times[-1] = start + ' ' + end
        else:
            start, end = times[-1].split()
            if float(end) < float(duration):
                labs.append('sp')
                times.append(end + ' ' + duration)
        # write lab files
        fp = open(dirout + '/' + uttid + '.lab', 'w')
        for i in range(len(labs)):
            fp.write("%s %s\n" % (times[i], labs[i].split('_')[0]))
        fp.close()
        
def write_as_wrdlabs(kaldialign, wavdurations, labdir, dirout):
    lastid = ''
    lasttime = 0.0
    lines = open(kaldialign).readlines()
    for l in lines:
        ll = l.split()
        if ll[0] != lastid:
            if lastid:
                labs = open(os.path.join(labdir, lastid + '.lab')).readlines()
                etime = float(labs[-1].split()[1])
                fp.write("%.3f %.3f SIL\n" % (lasttime, etime))
                fp.close()
            fp = open(dirout + '/' + ll[0] + '.wrd', 'w')
            lastid = ll[0]
            lasttime = 0.0
        stime = float(ll[2])
        etime = stime + float(ll[3])
        # output a silence symbol if there is a gap
        if "%.3f" % stime != "%.3f" % lasttime:
            fp.write("%.3f %.3f SIL\n" % (lasttime, stime))
        
        fp.write("%.3f %.3f %s\n" % (stime, etime, ll[4].lower()))
        lasttime = etime
    labs = open(os.path.join(labdir, lastid + '.lab')).readlines()
    etime = float(labs[-1].split()[1])
    fp.write("%.3f %.3f SIL\n" % (lasttime, etime))
    fp.close()

# TODO Should really use dom to write this XML more flexible for future
# xml structure etc.
# TODO given the silence duration and location it is possible to look at f0
# across recent phonetic material to decide if a break is a rising or falling
def write_xml_textalign(breaktype, breakdef, labdir, wrddir, output):
    fpx = open(output, 'w')
    fpx.write("<document>\n")
    #get files
    labs = glob.glob(labdir + '/*.lab')
    labs.sort()
    for l in labs:
        stem = os.path.splitext(os.path.split(l)[1])[0]
        fpx.write("<fileid id='%s'>" % stem)
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
                fpx.write("<lex pron='%s'>%s</lex> " % (' '.join(pron), ww[2]))
            else:
                if not widx:
                    fpx.write("<break type='%s'/>" % breakdef)
                elif widx == len(words) - 1:
                    fpx.write("<break type='%s'/>" % breakdef)
                else:
                    btype = breakdef
                    for b in breaktype.split(','):
                        bb = b.split(':')
                        minval = float(bb[1])
                        if float(ww[1]) - float(ww[0]) < minval:
                            btype = bb[0]
                    fpx.write("<break type='%s'/>" % btype)
        fpx.write("</fileid>\n")
    fpx.write("</document>\n")
    fpx.close()
                
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
    # copy setup data
    com = 'rm -rf %s' % (os.path.join(build_conf.outdir, 'output', 'data'))
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
                os.path.join(build_conf.kaldidir, 'src', 'latbin'),
                os.path.join(build_conf.kaldidir, 'src', 'idlakfexbin'),
                os.path.join(build_conf.kaldidir, 'src', 'gmmbin')]
    os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
    datadir = os.path.join(build_conf.outdir, 'output', 'data')
    # create lang directory using kaldi script
    com = "cd %s/output; utils/prepare_lang.sh data '<OOV>' data/lang data/lang" % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to build lang subdir')
    os.system(com)
    # extract mfccs
    com = "cd %s/output; steps/make_mfcc.sh --nj 1 data/train data/mfcc_log data/mfcc" % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to extract mfccs')
    os.system(com)
    # build dummy spk to utt file
    com = "cd %s/output; utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt" % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute dummy spk2utt file')
    os.system(com)
    # compute feature stats
    com = "cd %s/output; steps/compute_cmvn_stats.sh data/train data/mfcc_log data/mfcc" % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute feature statistics')
    os.system(com)
    # mono train
    com = 'cd %s/output; steps/train_mono.sh --nj 1 data/train data/lang kaldimono_output' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to compute flat start monophone models')
    os.system(com)
    # extract the phone alignment
    com = 'cd %s/output; show-alignments data/lang/phones.txt kaldimono_output/40.mdl "ark:gunzip -c kaldimono_output/ali.1.gz|" > align.dat' % (build_conf.outdir)
    build_conf.logger.log('info', 'running kaldi script to extract alignment')
    os.system(com)
    #extract the word alignment
    com = 'cd %s/output; linear-to-nbest "ark:gunzip -c kaldimono_output/ali.1.gz|" "ark:utils/sym2int.pl --map-oov 1669 -f 2- data/lang/words.txt < data/train/text |" \'\' \'\' ark:- | lattice-align-words data/lang/phones/word_boundary.int kaldimono_output/40.mdl ark:- ark:- | nbest-to-ctm ark:- - | utils/int2sym.pl -f 5 data/lang/words.txt > wrdalign.dat' % (build_conf.outdir)
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
    write_as_labs(os.path.join(build_conf.outdir, 'output', 'align.dat'), wavdurations, labdir)
    wrddir = os.path.join(build_conf.outdir, 'output', 'wrds')
    if not os.path.isdir(wrddir):
        os.mkdir(wrddir)
    write_as_wrdlabs(os.path.join(build_conf.outdir, 'output', 'wrdalign.dat'), wavdurations, labdir, wrddir)
    #write alignment based xml text file
    write_xml_textalign(breaktype, breakdef, labdir, wrddir, os.path.join(build_conf.outdir, 'output', 'text.xml'))
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
