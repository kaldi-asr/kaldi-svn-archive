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

# Full model context creation (linguistic context extraction)

import sys, os.path, time, subprocess, re, shlex
from xml.dom.minidom import parse, parseString

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Full model context creation (linguistic context extraction)'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

def ProduceLookupTable(xml_file):
    # Produce a lookup table which changes the sets (e.g. phones) into ints.
    # Save out so other applications can change the data back.
    lookup_table = {}

    dom = parse(xml_file)

    sets = dom.getElementsByTagName('set')

    for s in sets:
        set_name = s.getAttribute('name')
        set_items = s.getElementsByTagName('item')

        item_dict = {}
        lookup_table[set_name] = item_dict

        # We want to key the items based on their order in the file.
        item_id = 0
        for item in set_items:
                item_name = item.getAttribute('name')
                lookup_table[set_name][item_name] = item_id
                item_id = item_id + 1

    return lookup_table

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
    tpdbdir = os.path.join(build_conf.idlakdata, build_conf.lang, build_conf.acc)
    outdir = build_conf.outdir
    # get required directories from dependent modules
    aligndir = build_conf.get_input_dir('align_def')
    # examine modulespecific settings and set as appropriate
    # process data
    # get path to txpbin
    pathlist = [os.path.join(build_conf.kaldidir, 'src', 'idlaktxpbin')]
    os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
    # open the aligner xml output with minidom
    dom = parse(os.path.join(aligndir, "text.xml" ))
    normcmd = "idlaktxp --pretty --tpdb=%s - -" % (tpdbdir)
    normpipe = subprocess.Popen(shlex.split(normcmd.encode('utf8')), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    normtext = normpipe.communicate(input=dom.toxml())[0]
    normpipe.stdout.close()
    cexcmd = "idlakcex --pretty --tpdb=%s - -" % (tpdbdir)
    cexpipe = subprocess.Popen(shlex.split(cexcmd.encode('utf8')), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    cex = cexpipe.communicate(input=normtext)[0]
    cexpipe.stdout.close()

    xml_file = '../../idlak-data/en/ga/cex-default.xml'
    dom = parse(xml_file)
    
    # Convert the idlakcex output to integers using the lookup table.
    # We want a dict of the feature delimiters alongside the set which the feature belongs to [if any]
    feature_delimiters = {}

    features = dom.getElementsByTagName('feat')

    for f in features:
      current_delim = f.getAttribute('delim')
      current_set = f.getAttribute('set')
      feature_delimiters[current_delim] = current_set

    # Process 'normtext' with minidom into something which kaldi will understand.
    # Probably need to use the look-up table at this stage to change the phones to ints.
    lookup_table = ProduceLookupTable(xml_file)
    #print lookup_table

    dom = parseString(cex)

    fileids = dom.getElementsByTagName('fileid')

    output_filename = os.path.join(outdir, 'output', 'cex.dat')
    output_file = open(output_filename, 'w')

    for f in fileids:
        phons = f.getElementsByTagName('phon')

        for p in phons:
            cex_string = p.firstChild.nodeValue

            # We parse the idlakcex output to ascertain the values for each feature.
            escaped_delimiters = []

            # Need to escape non-alphanumeric characters in the delimiters.
            for k in feature_delimiters.keys():
                escaped_delimiters.append(re.escape(k))

            delimiter_string = '|'.join(escaped_delimiters)
            # Extract values and their delimiters, ?= indicates that we're looking for the 
            # next delimiter or the end of the string, but *not* consuming it.
            regex_string = "(%s)(\w+?)(?=%s|$)" % (delimiter_string, delimiter_string)

            pat = re.finditer(regex_string, cex_string)

            # This will hold the values of all the phone's features.
            feature_list = []

            for match in pat:
                # Check if this feature uses a set. If so, match.group(1) will name the set.
                # If it does, we need to convert the value to an integer.
                if feature_delimiters[match.group(1)] != '':
                    feature_value = lookup_table[feature_delimiters[match.group(1)]][match.group(2)]
                else:
                    feature_value = match.group(2)

                feature_list.append(str(feature_value))

            output_file.write('%s\n' % (' '.join(feature_list)))

    output_file.close()

    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)

if __name__ == '__main__':
    main()
    
