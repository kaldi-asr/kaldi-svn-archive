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

import sys, os.path, time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Builds trees based on alignments, context extraction data, and acoustic features'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

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
    com = '%s %s "ark:gunzip -c %s|" ark,t:%s ark:%s' % (makefullctx,
                                                         quinphonemodel,
                                                         quinphonealign,
                                                         contextdata,
                                                         fullctxali)
    os.system(com)
    # accumalate statistics for pitch
    fullctxacc = os.path.join(build_conf.kaldidir, 'src', 'bin',
                               'acc-fullctx-stats')
    pitchfeatures = os.path.join(pitchdir, 'output.ark')
    pitchacc = os.path.join(build_conf.outdir, 'output', 'pitch_acc.dat')
    com = '%s 2 ark:%s ark:%s %s' % (fullctxacc,
                                     pitchfeatures,
                                     fullctxali,
                                     pitchacc)
    print com
    os.system(com)
                                                
    
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
