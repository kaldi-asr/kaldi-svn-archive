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

# 

import sys, os.path, time, glob

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Performs mel-cepstrum analysis on given .wav files.'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

def load_input_wavs(wavdir, flist):
    valid_ids = {}
    wavs = glob.glob(wavdir + '/*.wav')
    for w in wavs:
        filename = os.path.splitext(os.path.split(w)[1])[0]

        if len(flist):
            if flist.has_key(filename):
                valid_ids[filename] = 1
        else:
            valid_ids[filename] = 1

    # If there are no flist files present in wavdir, we can't proceed.
    if not wavs:
        raise Exception("No files specified in flist are present in %s" % (wavdir))

    return valid_ids

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
    # get required directories from dependent modules
    kaldisrcdir = os.path.join(build_conf.kaldidir, 'src')
    # examine general settings and set as appropriate
    sptk_root = build_conf.getval('mcep_def', 'sptk_root')

    if not os.path.isdir(sptk_root):
        build_conf.logger.log('error', 'Supplied sptk_root location %s does not exist!' % (sptk_root))
        raise IOError('Supplied sptk_root location %s does not exist!' % (sptk_root))

    # process data
    wavdir = os.path.join(build_conf.idlakwav, build_conf.lang, build_conf.acc, build_conf.spk, build_conf.srate)

    valid_ids = load_input_wavs(wavdir, build_conf.flist)

    sptk_bin_root = os.path.join(sptk_root, 'bin')

    for wavfile in valid_ids:
        window_length = 400
        frame_shift = 80
        # all-pass constant
        alpha = 0.42
        # order of mel-generalised cepstrum
        order = 12

        # Strips headers from RIFF wav file.
        wavdata_com = '%s/featbin/wav-data %s/%s.wav' % (kaldisrcdir, wavdir, wavfile)
        # Converts data from short to float (+sf).
        x2x_com = '%s/x2x/x2x +sf' % (sptk_bin_root)
        frame_com = '%s/frame/frame -l %s -p %s' % (sptk_bin_root, window_length, frame_shift)
        # '-L 512' is the output frame length.
        # '-w 1' refers to the usage of a Hamming window.
        # '-n 1' is sigma(n=0,L-1)(w2(n)=1) normalisation.
        window_com = '%s/window/window -l %s -L 512 -w 1 -n 1' % (sptk_bin_root, window_length)
        # '-e 0.001' is a small value added to periodgram
        # '-l 512' is frame length.
        mcep_com = '%s/mcep/mcep -a %s -e 0.001 -m %s -l 512' % (sptk_bin_root, alpha, order)

        com = '%s | %s | %s | %s | %s | %s/x2x/x2x +fa' % (wavdata_com, x2x_com, frame_com, window_com, mcep_com, sptk_bin_root)
        build_conf.logger.log('info', com)
        com_output = os.system(com)

    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
