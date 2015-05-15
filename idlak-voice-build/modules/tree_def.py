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

# Builds trees based on alignments, context extraction data, and acoustic features

import sys, os.path, time, math

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Builds trees based on alignments, context extraction data, and acoustic features'
FRAMESHIFT = 0.005
NOSTATES = 5

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration




# process alignment to produce durations parameters from state alignments
# and full context phoness

# Apologies for hacking this in python instead of using proper kaldi methods
# expects alignment to be in text format. MA100315
# new alignment has a single value per phone with the state value removed
# durations parameters are dim no_states and have a value for each phone
def convert_ali_durations_data(no_states, ali, outputdir):
    statedurarkout = open(os.path.join(outputdir, 'durations_states.ark'), 'w')
    phonedurarkout = open(os.path.join(outputdir, 'durations_phones.ark'), 'w')
    aliout = open(os.path.join(outputdir, 'durationali.ark'), 'w')
    fp = open(ali)
    line = fp.readline()
    while(line):
        line = line.strip()
        frames = line.split(' ;')
        # get keyname out of first frame
        key = frames[0].split()[0]
        statedurarkout.write(key + ' [\n')
        phonedurarkout.write(key + ' [\n')
        aliout.write(key)
        frames[0] = ' '.join(frames[0].split()[1:])
        last_state = -1
        last_frame = ''
        last_state_sz = 0
        durations = []
        state_szs = []
        # HTS expects all models to be 5 state in kaldi silence
        # models are not. So fake a 5ms state if not present.
        for i in range(no_states): state_szs.append(1.0)
        for f in frames[:-1]:
            f = f.strip()
            ff = f.split()
            state = int(ff[-1])
            # state sizes
            if last_state > -1 and state != last_state:
                state_szs[last_state] = float(last_state_sz)
                last_state_sz = 0
            # output last phone data
            if last_state > state:
                phonedur = 0
                for v in state_szs:
                    phonedur += v
                    statedurarkout.write(' %f' % (v))
                statedurarkout.write('\n')
                phonedurarkout.write(' %f\n' % phonedur)
                aliout.write(' ' + last_frame +' ;')
                # reset state sizes
                for i in range(no_states): state_szs[i] = 1.0
            last_state = state
            # zero state value
            last_frame = ' '.join(ff[:-1]) + ' 0'
            last_state_sz += 1
        state_szs[last_state] = float(last_state_sz)
        phonedur = 0
        for v in state_szs:
            phonedur += v
            statedurarkout.write(' %f' % (v))
        statedurarkout.write(' ]\n')
        phonedurarkout.write(' %f ]\n' %  phonedur)
        aliout.write(' ' + last_frame + ' ;\n')
        line = fp.readline()
    statedurarkout.close()
    phonedurarkout.close()
    aliout.close()

def main():
    # process the options based on the default build configuration
    build_conf, parser = build_configuration.get_config(SCRIPT_NAME, DESCRIPTION, SCRIPT_NAME)
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

    # MODULE SPECIFIC CODE
    # get required input files from idlak-data
    # question file
    
    # get required directories from dependent modules
    aligndir = build_conf.get_input_dir('align_def')
    cexdir = build_conf.get_input_dir('cex_def')
    pitchdir = build_conf.get_input_dir('pitch_def')
    outdir_data = os.path.join(build_conf.outdir, 'data')
    if not os.path.isdir(outdir_data):
        os.mkdir(outdir_data)

    # examine general settings and set as appropriate
    # process data
    # merge full context alignment with quinphone alignment
    build_conf.logger.log('info', 'Merging full context information with quinphone alignment')
    makefullctx = os.path.join(build_conf.kaldidir, 'src', 'bin',
                               'make-fullctx-ali')
    quinphonemodel = os.path.join(aligndir, 'kaldidelta_quin_output',
                                  'final.mdl')
    quinphonealign = os.path.join(aligndir, 'kaldidelta_quin_output',
                                  'ali.1.gz')
    contextdata = os.path.join(cexdir, 'cex.ark')
    fullctxali = os.path.join(build_conf.outdir, 'output', 'ali')
    com = '%s %s "ark:gunzip -c %s|" ark,t:%s ark,t:%s' % (makefullctx,
                                                         quinphonemodel,
                                                         quinphonealign,
                                                         contextdata,
                                                         fullctxali)
    os.system(com)
    # compile context question sets from cex_def
    compilequestions = os.path.join(build_conf.kaldidir, 'src', 'bin',
                                    'compile-questions')
    ctxqset = os.path.join(cexdir, 'qset.dat')
    # dummy questions.int
    dummyqset = os.path.join(build_conf.outdir, 'output', 'questions.int')
    os.system("touch %s" % (dummyqset))
    topo =  os.path.join(aligndir, 'data', 'lang', 'topo')
    ctxqsetbin = os.path.join(build_conf.outdir, 'output', 'qset_binary.dat')
    # unclear how the topology affects the pdf generation here
    com = "%s --central-position=2 --binary=false --context-width=5 --keyed-questions=%s %s %s %s" % (compilequestions,
                                                                                       ctxqset,
                                                                                       topo,
                                                                                       dummyqset,
                                                                                       ctxqsetbin)
    os.system(com)
    
    # accumulate statistics for pitch
    fullctxacc = os.path.join(build_conf.kaldidir, 'src', 'bin',
                               'acc-fullctx-stats')
    pitchfeatures = os.path.join(pitchdir, 'lf0.ark')
    pitchacc = os.path.join(build_conf.outdir, 'output', 'pitch_acc.dat')
    com = '%s --binary=false 2 ark:%s ark:%s %s' % (fullctxacc,
                                                    pitchfeatures,
                                                    fullctxali,
                                                    pitchacc)
    os.system(com)

    # build a tree
    buildtree = os.path.join(build_conf.kaldidir, 'src', 'bin',
                             'build-tree')
    roots = os.path.join(aligndir, 'data', 'lang', 'phones', 'roots.int')
    rootsdummy = '/afs/inf.ed.ac.uk/user/m/matthewa/kaldi/matthewa/kaldi-idlak/idlak-voice-build/dummy.int'
    treeout = os.path.join(build_conf.outdir, 'output', 'pitch.tree')
    com = "%s --binary=false --verbose=1 --context-width=5 --central-position=2 %s %s %s %s %s" % (buildtree,
                                                                                    pitchacc,
                                                                                    roots,
                                                                                    ctxqsetbin,
                                                                                    topo,
                                                                                    treeout)
    os.system(com)

    # make a model from the tree and the statistics
    gmminitmodel = os.path.join(build_conf.kaldidir, 'src', 'gmmbin',
                                                 'gmm-init-model')
    modelout = os.path.join(build_conf.outdir, 'output', 'pitch.mdl')
    com = "%s --binary=false %s %s %s %s" % (gmminitmodel, treeout, pitchacc, topo, modelout)
    os.system(com)


    ######################################################################
    #                  DURATION MODELLING
    ######################################################################                  
    # modify full context alignment to have a single line for each phone
    # and generate duration parameters for state durations
    convert_ali_durations_data(NOSTATES, fullctxali, outdir_data)

    # build context question set
    ctxqsetbin = os.path.join(build_conf.outdir, 'output', 'qset_binary_dur.dat')
    com = "%s --central-position=2 --binary=false --context-width=5 --keyed-questions=%s %s %s %s" % (compilequestions,
                                                                                       ctxqset,
                                                                                       topo + '2',
                                                                                       dummyqset,
                                                                                       ctxqsetbin)
    os.system(com)
    
    # accumulate statistics for state duration
    stateduracc = os.path.join(build_conf.outdir, 'output', 'statedur_acc.dat')
    com = '%s --binary=false --var-floor=20.0 2 ark:%s ark:%s %s' % (fullctxacc,
                                                    os.path.join(outdir_data, 'durations_states.ark'),
                                                    os.path.join(outdir_data, 'durationali.ark'),
                                                    stateduracc)
    os.system(com)
    # accumulate statistics for phone durations
    phoneduracc = os.path.join(build_conf.outdir, 'output', 'phonedur_acc.dat')
    com = '%s --binary=false --var-floor=20.0 2 ark:%s ark:%s %s' % (fullctxacc,
                                                    os.path.join(outdir_data, 'durations_phones.ark'),
                                                    os.path.join(outdir_data, 'durationali.ark'),
                                                    phoneduracc)
    os.system(com)
    
    # build a tree
    # For Interspeech 15 work we have the following duration trees and models
    # 1. Kaldi out of the box
    # 2. Kaldi with same number of leaves and no initial roots questions
    # 3. As 2 but using 5 dim state duration data
    # 4. As 3 but with no post processing
    treeout1 = os.path.join(build_conf.outdir, 'output', 'dur_1.tree')
    treeout2 = os.path.join(build_conf.outdir, 'output', 'dur_2.tree')
    treeout3 = os.path.join(build_conf.outdir, 'output', 'dur_3.tree')
    treeout4 = os.path.join(build_conf.outdir, 'output', 'dur_4.tree')
    #11.3 thresh for statedur stats -> 511 leaves
    #8.3 thresh for phonedur stats -> 518 leaves
    com = "%s --binary=false --verbose=1 --context-width=5 --central-position=2 %s %s %s %s %s" % (buildtree,
                                                                                    phoneduracc,
                                                                                    roots,
                                                                                    ctxqsetbin,
                                                                                    topo + '2',
                                                                                    treeout1)
    os.system(com)
    com = "%s --binary=false --max-leaves=513 --thresh=0 --verbose=1 --context-width=5 --central-position=2 %s %s %s %s %s" % (buildtree,
                                                                                    phoneduracc,
                                                                                    rootsdummy,
                                                                                    ctxqsetbin,
                                                                                    topo + '2',
                                                                                    treeout2)
    os.system(com)
    com = "%s --binary=false --max-leaves=513  --thresh=0 --verbose=1 --context-width=5 --central-position=2 %s %s %s %s %s" % (buildtree,
                                                                                    stateduracc,
                                                                                    rootsdummy,
                                                                                    ctxqsetbin,
                                                                                    topo,
                                                                                    treeout3)
    os.system(com)
    com = "%s --binary=false --max-leaves=513  --cluster-thresh=0 --thresh=0 --verbose=1 --context-width=5 --central-position=2 %s %s %s %s %s" % (buildtree,
                                                                                    stateduracc,
                                                                                    rootsdummy,
                                                                                    ctxqsetbin,
                                                                                    topo,
                                                                                    treeout4)
    os.system(com)
    
    # make a model from the tree and the state statistics
    modelout1 = os.path.join(build_conf.outdir, 'output', 'dur_1.mdl')
    modelout2 = os.path.join(build_conf.outdir, 'output', 'dur_2.mdl')
    modelout3 = os.path.join(build_conf.outdir, 'output', 'dur_3.mdl')
    modelout4 = os.path.join(build_conf.outdir, 'output', 'dur_4.mdl')
    com = "%s --binary=false %s %s %s %s" % (gmminitmodel, treeout1, stateduracc, topo, modelout1)
    os.system(com)
    com = "%s --binary=false %s %s %s %s" % (gmminitmodel, treeout2, stateduracc, topo, modelout2)
    os.system(com)
    com = "%s --binary=false %s %s %s %s" % (gmminitmodel, treeout3, stateduracc, topo, modelout3)
    os.system(com)
    com = "%s --binary=false %s %s %s %s" % (gmminitmodel, treeout4, stateduracc, topo, modelout4)
    os.system(com)
    
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
        
if __name__ == '__main__':
    main()
    
