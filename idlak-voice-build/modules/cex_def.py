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
    qset =  os.path.join(build_conf.idlakdata, build_conf.lang, build_conf.acc, "qset-default.xml")
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
    # keep a copy of the original xml
    output_filename = os.path.join(outdir, 'output', 'cex.xml')
    output_file = open(output_filename, 'w')
    output_file.write(cex)
    
    dom = parseString(cex)
    # get header information
    header = dom.getElementsByTagName('txpheader')[0]
    cexheader = header.getElementsByTagName('cex')[0]
    
    # get by file ids
    fileids = dom.getElementsByTagName('fileid')

    output_filename = os.path.join(outdir, 'output', 'cex.dat')
    output_file = open(output_filename, 'w')

    # output the contexts only and build a string to integer table
    # this assumes space delimited contexts
    # note phone context is handled separately and the first
    # set of contexts (non space delimited) are phones
    # The current phone is prepended to the other contexts
    # to keep track of silences which may differ at start and end of
    # utterenaces from the alignment as it stands.
    # note data is also stored to be reformatted into kaldi in output_contexts
    freqtables = {}
    output_contexts = []
    for f in fileids:
        phons = f.getElementsByTagName('phon')
        output_contexts.append([f.getAttribute('id'), []])
        last_phon_name = ''
        for p in phons:
            phon_name = p.getAttribute('val')
            # Currently ignore utt internal split pauses
            if phon_name == 'pau' and last_phon_name == 'pau':
                last_phon_name = phon_name
                continue
            cex_string = p.firstChild.nodeValue
            cexs = cex_string.split()[1:]
            # get context phone name (may be different to xml phon val)
            pat = re.match('\^(.*?)\~(.*?)\-(.*?)\+(.*?)\=(.*)', cex_string.split()[0])
            if not pat:
                build_conf.logger.log('critical', 'bad phone context string %s %s' %
                                      (f, cex_string.split()[0]))
            phonename = pat.group(3)
            # currently add phone contexts as first 5 features
            # this to avoid a mismatch between manual phone
            # questions and the kaldi context information
            cexs = list(pat.groups()) + cexs
            # Currently set all contexts in pause to 0
            if phon_name == 'pau':
                for i in range(len(cexs)): cexs[i] = '0'
            # prepend the phone to keep track of silences and for sanity checks
            cexs.insert(0, phonename)
            # save/write contexts
            output_file.write('%s %s\n' % (f.getAttribute('id'), ' '.join(cexs)))
            output_contexts[-1][-1].append(cexs)
            # keep track of context frequencies
            for i in range(len(cexs)):
                key = 'cex' + ('000' + str(i))[-3:]
                if not freqtables.has_key(key):
                    freqtables[key] = {}
                if not freqtables[key].has_key(cexs[i]):
                    freqtables[key][cexs[i]] = 1
                else:
                    freqtables[key][cexs[i]] += 1
            last_phon_name = phon_name

    output_file.close()

    # write frequency tables of contexts for audit purposes
    for ftable in freqtables.keys():
        fp = open(os.path.join(outdir, 'output', ftable + '_freq.txt'), 'w')
        vals = freqtables[ftable].keys()
        vals.sort()
        for v in vals:
            fp.write("%s %d\n" % (v, freqtables[ftable][v]))
        fp.close()
        
    # create lookup tables if required
    lookuptables = {}
    for i in range(len(cexs)):
        key = 'cex' + ('000' + str(i))[-3:]
        vals = freqtables[key].keys()
        vals.sort()
        for v in vals:
            if not re.match('[0-9]+', v):
                # found a non integer value create a lookup table
                lookuptables[key] = {}
                mapping = 1
                for v in vals:
                    if v == '0':
                        lookuptables[key][v] = 0
                    else:
                        lookuptables[key][v] = mapping
                        mapping += 1
                break
            
    # output lookup tables
    for table in lookuptables.keys():
        fp = open(os.path.join(outdir, 'output', table + '_lkp.txt'), 'w')
        vals = lookuptables[table].keys()
        vals.sort()
        for v in vals:
            fp.write("%s %d\n" % (v, lookuptables[table][v]))
        fp.close()

    # write kaldi style archive replacing symbols with lookup
    output_filename = os.path.join(outdir, 'output', 'cex.ark')
    fp = open(output_filename, 'w')
    for f in output_contexts:
        key = f[0]
        fp.write(key + ' ')
        for p in f[1]:
            for i, v in enumerate(p):
                # replace symbols with integers
                table = 'cex' + ('000' + str(i))[-3:]
                if lookuptables.has_key(table):
                    v = str(lookuptables[table][v])
                fp.write(v + ' ')
            fp.write('; ')
        fp.write('\n')
    fp.close()
    
    # create kaldi style question set
    # index of features is +5 (quin phone context) -1 (phone prepended to cex output)
    kaldiqset = open(os.path.join(outdir, 'output', 'qset.dat'), 'w')
    qsetxml = parse(qset)
    contexts = cexheader.getAttribute('cexfunctions').split(';')
    for i, c in enumerate(contexts):
        for qcontext in qsetxml.getElementsByTagName('feat'):
            if qcontext.getAttribute('name') == c:
                for qs in qcontext.getElementsByTagName('qs'):
                    qindex = i + 5
                    vals = (qs.firstChild.nodeValue).strip().split(' ')
                    # replace vals with integers using lookup table if required
                    table = 'cex' + ('000' + str(i + 1))[-3:]
                    if lookuptables.has_key(table):
                        newvals = []
                        errors = []
                        for j in range(len(vals)):
                            if not lookuptables[table].has_key(vals[j]):
                                errors.append(vals[j])
                            else:
                                newvals.append(str(lookuptables[table][vals[j]]))
                        vals = newvals
                        if len(errors):
                            build_conf.logger.log('warn', 'qs:%s %s not in database' % (qs.getAttribute('name'), str(errors)))
                            # print lookuptables[table], errors
                    if len(vals):
                        kaldiqset.write("%d ? %s\n" % (qindex, ' '.join(vals)))
    kaldiqset.close()                
            
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)

if __name__ == '__main__':
    main()
    
