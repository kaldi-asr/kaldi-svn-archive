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

# based on the delimiters unpacks the cex string into a list
def cex2list(cexheader, cex):
    vals = []
    predelim = ''
    cex = cex.strip()
    for idx, cexfunction in enumerate(cexheader.getElementsByTagName('cexfunction')):
        if not idx:
            predelim = cexfunction.getAttribute('delim')
            cex = cex[len(predelim):]
        else:
            pstdelim = cexfunction.getAttribute('delim')
            vals.append(cex[0:cex.find(pstdelim)])
            cex = cex[cex.find(pstdelim) + len(pstdelim):]
    vals.append(cex)
    return vals
    
def checkfilecontext(ref, cex, badctx):
    for i in range(len(ref)):
        if ref[i] != cex[i] and i not in badctx:
            badctx.append(i)


def badfilecontext2htsregex(cexheader, badfilecontexts):
    result = []
    cexfunctions = cexheader.getElementsByTagName('cexfunction')
    for idx, cexfunction in enumerate(cexfunctions):
        if idx in badfilecontexts:
            continue
        predelim = cexfunction.getAttribute('delim')
        pstdelim = ''
        if idx + 1 < len(cexfunctions):
            pstdelim = cexfunctions[idx + 1].getAttribute('delim')
        result.append(predelim + '.*' + pstdelim)
    return result
            
def output_kaldicex(logger, dom, outdir):
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
                logger.log('critical', 'bad phone context string %s %s' %
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
    return cexs, output_contexts, freqtables, cexheader

def output_htscex(logger, dom, outdir, phon_labs):
    # get header information
    header = dom.getElementsByTagName('txpheader')[0]
    cexheader = header.getElementsByTagName('cex')[0]
    # get by file ids
    fileids = dom.getElementsByTagName('fileid')
    # create hts lab directory if required
    htslabdir = os.path.join(outdir, 'output', 'htslab')
    if not os.path.exists(htslabdir):
        os.mkdir(htslabdir)
    # output the contexts only
    # we need to determine whch feature to use for
    # for HTS style parallel processing. It has to be the same within
    # each files
    badfilecontexts = []
    for f in fileids:
        fileid = f.getAttribute('id')
        for idx, spt in enumerate(f.getElementsByTagName('spt')):
            sptnostr = ('000' + str(idx))[-3:]
            start, end, labs = phon_labs["%s_%s" % (fileid, sptnostr)]
            phons = spt.getElementsByTagName('phon')
            fp = open(os.path.join(htslabdir, fileid + '_' + sptnostr + '.lab'), 'w')
            last_phon_name = ''
            for idx2, p in enumerate(phons):
                phon_name = p.getAttribute('val')
                cex_string = p.firstChild.nodeValue
                cexl = cex2list(cexheader, cex_string)
                if not idx2:
                    refcex = cexl
                checkfilecontext(refcex, cexl, badfilecontexts)
                fp.write("%d %d %s\n" % (int(labs[idx2][0] * 10000000),
                                         int(labs[idx2][1] * 10000000),
                                         cex_string))
            fp.close()
    return badfilecontext2htsregex(cexheader, badfilecontexts), cexheader

# write a file giving information for chopping up original waveforms
# into spurts (silence separated chunks of speech)
def write_spt_times(logger, dom, wrdsdir, output):
    # get by file ids
    fileids = dom.getElementsByTagName('fileid')
    start = 0.0
    fp = open(output, 'w')
    spt_labs = {}
    for f in fileids:
        fileid = f.getAttribute('id')
        labs = open(os.path.join(wrdsdir, fileid + '.lab')).readlines()
        #force duration zero silences at beginning and end
        if labs[0].split()[2] !='sil':
            labs.insert(0, "0.000 0.000 sil\n")
        if labs[-1].split()[2] !='sil':
            end = labs[-1].split()[1]
            labs.append(end + " " + end + " sil\n")
        
        sptno = 0
        newlabs = []
        totspts = len(f.getElementsByTagName('spt'))
        start = 0.0
        for idx, l in enumerate(labs):
            ll = l.split()
            if idx == 0:
                if ll[2] == 'sil' and float(ll[1]) > 0.100:
                    start = float(ll[1]) - 0.100
                newlabs.append([0.0, float(ll[1]) - start, 'sil'])
            elif ll[2] != 'sil':
                newlabs.append([float(ll[0]) - start, float(ll[1]) - start, ll[2]])
            if idx > 0 and ll[2] == 'sil':
                dur = float(ll[1]) - float(ll[0])
                if dur < 0.200:
                    end = float(ll[0]) + 0.5 * dur
                    newstart = float(ll[0]) + 0.5 * dur
                else:
                    end = float(ll[0]) + 0.100
                    newstart = float(ll[1]) - 0.100
                sptnostr = ('000' + str(sptno))[-3:]
                newlabs.append([float(ll[0]) - start, end - start, 'sil'])
                # should be a kaldi binary specified here MA050314
                fp.write("%s_%s.wav %.3f %.3f %s.wav\n" % (fileid,
                                                           sptnostr,
                                                           start,
                                                           end,
                                                           fileid))
                spt_labs["%s_%s" % (fileid, sptnostr)] = [start, end, newlabs]
                start = newstart
                sptno += 1
                newlabs = [[0.0, float(ll[1]) - start, 'sil']]
        if totspts != sptno:
            print labs
            logger.log('warn', 'mismatch in spt numbers: %s (xml)%d vs (lab)%d' % (fileid,
                                                                                    totspts,
                                                                                    sptno))
    fp.close()
    return spt_labs              
        
def write_kaldiqset(logger, defqset, kaldiqsetfile, cexheader, lookuptables):
    # create kaldi style question set
    kaldiqset = open(kaldiqsetfile, 'w')
    # index of features is +5 (quin phone context) -1 (phone prepended to cex output)
    qsetxml = parse(defqset)
    contexts = []
    for cexfunction in cexheader.getElementsByTagName('cexfunction'):
        contexts.append(cexfunction.getAttribute('name'))
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
                                # check silence phone ('0' in kaldi always) map from 'sil' and 'pau'
                                if vals[j] == 'sil' or vals[j] == 'pau':
                                    newvals.append('0')
                                else:
                                    errors.append(vals[j])
                            else:
                                newvals.append(str(lookuptables[table][vals[j]]))
                        vals = newvals
                        if len(errors):
                            logger.log('warn', 'qs:%s %s not in database' % (qs.getAttribute('name'), str(errors)))
                            # print lookuptables[table], errors
                    if len(vals):
                        kaldiqset.write("%d ? %s\n" % (qindex, ' '.join(vals)))
    kaldiqset.close()                
            
def write_htsqset(logger, defqset, htsqsetfile, cexheader):
    # create kaldi style question set
    # index of features is +5 (quin phone context) -1 (phone prepended to cex output)
    htsqset = open(htsqsetfile, 'w')
    qsetxml = parse(defqset)
    contexts = []
    delims = []
    for cexfunction in cexheader.getElementsByTagName('cexfunction'):
        contexts.append(cexfunction.getAttribute('name'))
        delims.append(cexfunction.getAttribute('delim'))
    for i, c in enumerate(contexts):
        for qcontext in qsetxml.getElementsByTagName('feat'):
            if qcontext.getAttribute('name') == c:
                for qs in qcontext.getElementsByTagName('qs'):
                    predelim = delims[i]
                    if i > 0: predelim = '*' + predelim
                    pstdelim = ''
                    qsname = qs.getAttribute('name')
                    if i + 1 < len(delims):
                        pstdelim = delims[i + 1]
                    if pstdelim: pstdelim = pstdelim + '*'
                    vals = (qs.firstChild.nodeValue).strip().split(' ')
                    for j in range(len(vals)):
                        vals[j] = predelim + vals[j] + pstdelim
                    if len(vals):
                        htsqset.write('QS "%s" {%s}\n' % (qsname, ','.join(vals)))
    htsqset.close()                
            
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
    # get audio directory
    wavdir = os.path.join(build_conf.idlakwav, build_conf.lang, build_conf.acc,
                          build_conf.spk, build_conf.srate)
    if not os.path.isabs(wavdir):
        wavdir = os.path.realpath(os.path.join(os.path.curdir, wavdir))
    # get required directories from dependent modules
    aligndir = build_conf.get_input_dir('align_def')
    # Check to see if we generate HTS style context models as well
    hts = build_conf.getval('cex_def', 'hts')
    # examine modulespecific settings and set as appropriate
    # process data
    # get path to txpbin
    pathlist = [os.path.join(build_conf.kaldidir, 'src', 'idlaktxpbin')]
    os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
    # Process script through txp and cex
    output_filename = os.path.join(outdir, 'output', 'cex.xml')
    cmd = "idlaktxp --pretty --tpdb=%s %s - | " % (tpdbdir, os.path.join(aligndir, "text.xml")) + \
        "idlakcex --pretty --tpdb=%s - %s" % (tpdbdir, output_filename)
    os.system(cmd)
    # read in the cex xml output and generate kaldi files for tree building
    dom = parse(output_filename)
    cexs, output_contexts, freqtables, cexheader = output_kaldicex(build_conf.logger, dom, outdir)
    # write out script to split original wavs into spts if required (i.e for HTS test)
    phon_labs = write_spt_times(build_conf.logger,
                                dom,
                                os.path.join(aligndir, 'labs'),
                                os.path.join(outdir, 'output', 'spt_times.dat'))    
    # generate HTS style context model names
    if hts == "True":
        output_filename = os.path.join(outdir, 'output', 'cex_hts.xml')
        cmd = "idlaktxp --pretty --tpdb=%s %s - | " % (tpdbdir, os.path.join(aligndir, "text.xml")) + \
            "idlakcex --pretty --cex-arch=hts --tpdb=%s - %s" % (tpdbdir, output_filename)
        os.system(cmd)
        dom = parse(output_filename)
        filecontexts, cexheader = output_htscex(build_conf.logger, dom, outdir, phon_labs)
        htsqset = os.path.join(outdir, 'output', 'questions-kaldi-%s-%s.hed' % (build_conf.lang, build_conf.acc))
        write_htsqset(build_conf.logger, qset, htsqset, cexheader)

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
    
    kaldiqset = os.path.join(outdir, 'output', 'qset.dat')
    write_kaldiqset(build_conf.logger, qset, kaldiqset, cexheader, lookuptables)
    print filecontexts
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
