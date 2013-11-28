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

# Calculates the f0 from .wav files'

import sys, os.path, time, glob, math

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Calculates the f0 from .wav files'

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

def create_params_file(params_file, frame_step, minf0, maxf0):
    f = open(params_file, 'w')

    f.write('float frame_step = %s;' % (frame_step))

    if minf0 is not None:
        f.write('\nfloat min_f0 = %s;' % (minf0))
    if maxf0 is not None:
        f.write('\nfloat max_f0 = %s;' % (maxf0))

    f.close()

def run_f0_pass(wavdir, outdir, getf0_path, pplain_path, params_file, input_files):
    # Keep f0s for analysis
    f0s = []
    input_files = input_files.keys()
    input_files.sort()
    for f in input_files:
        f = f.strip()
        infile = os.path.join(wavdir, f + ".wav")
        outfilebin = os.path.join(outdir, f + ".f0.bin")
        outfile = os.path.join(outdir, f + ".f0")

        # Get f0 - output is in esps binary format (required for pitchmarking)
        # Also convert to text for voice generation

        if not params_file:
            f0_options = ''
        else:
            f0_options = '-P %s' % (params_file)

        com = '%s %s %s %s' % (getf0_path, f0_options, infile, outfilebin)
        print com

        retval = os.system(com)
        if retval:
            raise IOError("failed system command '%s'" % com)

        # Convert output to ascii
        # First two columns of output are required - f0, prob-voicing
        com = '%s %s' % (pplain_path, outfilebin)
        print com
        stdin, stdout, stderr = os.popen3(com)
        stdin.close()
        fp = open(outfile, 'w')
        fp.write("[ ");
        # HACK get_f0 seems to lose 2 frames at 5ms add a zero vlaue at front
        # and back
        fp.write("%s %s\n" % (0.0, 0.0))
        for l in stdout.readlines():
            firstcol = l.strip().split()[0]
            secondcol = l.strip().split()[1]
            # Getting it into kaldi ark format.
            # interpolate-pitch wants input in the form prob-voicing, f0.
            fp.write("%s %s\n" % (secondcol, firstcol))
            f0val = float(firstcol)
            if f0val > 0:
                f0s.append(float(firstcol))
        # HACK get_f0 seems to lose 2 frames at 5ms add a zero vlaue at front
        # and back
        fp.write("%s %s\n" % (0.0, 0.0))
        fp.write("]");
        stdout.close()
        fp.close()
        err = stderr.readlines()
        for l in err:
            l = l.strip()
            #if l:
                #raise IOError("error message from pplain: '%s'" % l)
        stderr.close()

    return f0s

def interpolate_f0s(kaldisrcdir, valid_ids, indir, outdir):
    # The purpose of this function is to interpolate a series of f0s,
    # removing the 0 instances.
    scp_filename=os.path.join(outdir, 'data', 'interpolate-f0s.scp')
    scp_file = open(scp_filename, 'w');

    outfile = os.path.join(outdir, 'output', 'output.ark')

    # Create scp file.
    for f in valid_ids:
        infile = os.path.join(indir, f + ".f0")

        scp_line = '%s %s/bin/copy-matrix %s - |\n' % (f, kaldisrcdir, infile)
        scp_file.write(scp_line)

    scp_file.close()

    # rspecifier and wspecifier are as described in /src/util/kaldi-table.h  
    rspecifier = 'scp:%s' % (scp_filename)
    wspecifier = 'ark,t:%s' % (outfile) 

    com = '%s/featbin/interpolate-pitch %s %s' % (kaldisrcdir, rspecifier, wspecifier)
    print com
    os.system(com)

def process_data(outdir, wavdir, getf0_path, pplain_path, flist, kaldisrcdir, force=False):
    valid_ids = load_input_wavs(wavdir, flist)

    # Stop ESPS writing configuration files that
    # prevent voice building in parallel
    os.putenv("USE_ESPS_COMMON", "off")

    # We want 5ms frames.
    frame_step = 0.005

    # Setup pass 1
    pass1_outdir = os.path.join(outdir, 'data', 'pass1')
    if not os.path.isdir(pass1_outdir):
        os.mkdir(pass1_outdir)

    # Create initial params file.
    params_file = os.path.join(pass1_outdir, 'params')
    create_params_file(params_file, frame_step, None, None)

    # Returns only non-zero (voiced values)
    f0s = run_f0_pass(wavdir, pass1_outdir, getf0_path, pplain_path, params_file, valid_ids)

    # We now want to log the values and find the +/-3 std. devs boundary.
    f0s_log = []
    for f0 in f0s:
        f0s_log.append(math.log(f0))

    f0s_log.sort()

    mean = sum(f0s_log)/float(len(f0s_log))

    sum_of_diffs_sqd = 0

    for f0 in f0s_log:
        sum_of_diffs_sqd += math.pow(f0-mean,2)

    std_dev = math.sqrt(sum_of_diffs_sqd/float(len(f0s_log)))

    f0_min_log = mean - (3*std_dev)
    f0_max_log = mean + (3*std_dev)

    f0_min = math.exp(f0_min_log)
    f0_max = math.exp(f0_max_log)

    # Setup pass 2
    pass2_outdir = os.path.join(outdir, 'data', 'pass2')
    if not os.path.isdir(pass2_outdir):
        os.mkdir(pass2_outdir)

    # Now recreate the params file with these new values for min/max.
    params_file = os.path.join(pass2_outdir, 'params')
    create_params_file(params_file, frame_step, f0_min, f0_max)

    # Run a second pass of get_f0/pplain.
    run_f0_pass(wavdir, pass2_outdir, getf0_path, pplain_path, params_file, valid_ids)

    # Interpolate in gaps.
    interpolate_f0s(kaldisrcdir, valid_ids, pass2_outdir, outdir)

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
    outdir = build_conf.outdir
    # get required directories from dependent modules
    kaldisrcdir = os.path.join(build_conf.kaldidir, 'src')
    # examine general settings and set as appropriate
    getf0_path = build_conf.getval('pitch_def', 'getf0')
    pplain_path = build_conf.getval('pitch_def', 'pplain')

    if not os.path.isfile(getf0_path):
        build_conf.logger.log('error', 'Supplied get_f0 location %s does not exist!' % (getf0_path))
        raise IOError('Supplied get_f0 location %s does not exist!' % (getf0_path))
    if not os.path.isfile(pplain_path):
        build_conf.logger.log('error', 'Supplied pplain location %s does not exist!' % (pplain_path))
        raise IOError('Supplied pplain location %s does not exist!' % (pplain_path))

    # process data
    wavdir = os.path.join(build_conf.idlakwav, build_conf.lang, build_conf.acc, build_conf.spk, build_conf.srate)
    
    outdir_data = os.path.join(outdir, "data")
    if not os.path.isdir(outdir_data):
        os.mkdir(outdir_data)

    process_data(outdir, wavdir, getf0_path, pplain_path, build_conf.flist, kaldisrcdir, True)
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
