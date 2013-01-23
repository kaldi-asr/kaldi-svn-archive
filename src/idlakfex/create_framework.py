# This python script takes the feature function code and the distance
# metric code and autogenerates a couple of C files to allow the
# functions to be selected run time.

import os, sys, string, re, glob, time

def get_fx_details(srcdir):
    files = glob.glob(os.path.join(srcdir, "fx", "fx*.c"))
    files.sort()
    feat_details = []
    for f in files:
        print f
	fname = os.path.split(f)[1]
        pat = re.match("fx_(.*)\.c", fname)
        fxname = pat.group(1)
        if fxname == "catalog":
            continue
        prototype = ""
        fwk = ""
        lines = open(f).readlines()
        for l in lines:
            #match function heading
            pat = re.match("(void\s+fx_" + fxname + ".*)\s*{.*", l)
            if pat:
                prototype = pat.group(1)
            pat = re.match("/\*\s+CPRC_FX_FRAMEWORK\s+(.*)\*/.*", l)
            if pat:
                fwk = pat.group(1).split()
        feat_details.append([fxname, prototype, fwk])
    return feat_details

def generate_fx_cmake(srcdir, feat_details):
    fp = open(os.path.join(srcdir, "fx", "fx.cmake"), 'w')
    fp.write("# Automatically generated: " + time.asctime() + "\n")
    fp.write("INCLUDE_DIRECTORIES(${MY_SOURCE_DIR}/fx)\nSET(CEREVOICE_FX\n")
    fp.write("\t\tfx/fx_catalog.c\n")
    for feat in feat_details:
        fp.write("\t\tfx/fx_" + feat[0] + ".c\n")
    fp.write(")\n")
    fp.close()
    
def generate_fx_catalog_h(srcdir, feat_details):
    fp = open(os.path.join(srcdir, "fx", "CPRC_fx_catalog.h"), 'w')
    fp.write(FILE_HEADER)
    fp.write(" " + time.asctime() + "*/\n")
    fp.write(FX_CATALOG_H_HEAD)
    fp.write(" " + str(len(feat_details)) + "\n\n")
    for feat in feat_details:
        fp.write(feat[1] + ";\n")
    fp.write(FX_CATALOG_H_TAIL)
    fp.close()

def generate_fx_catalog_c(srcdir, feat_details):
    fp = open(os.path.join(srcdir, "fx", "fx_catalog.c"), 'w')
    fp.write(FILE_HEADER)
    fp.write(" " + time.asctime() + "*/\n")
    fp.write(FX_CATALOG_C_HEAD)

    fp.write(FX_CATALOG_C_LINES["lbls"])
    for feat in feat_details[:-1]:
        fp.write('"' + feat[0] + '", ')
    fp.write('"' + feat_details[-1][0] + '"};\n')
    
    fp.write(FX_CATALOG_C_LINES["runbuild"])
    for feat in feat_details[:-1]:
        fp.write('' + feat[2][1] + ', ')
    fp.write(feat_details[-1][2][1] + '};\n')

    fp.write(FX_CATALOG_C_LINES["type"])
    for feat in feat_details[:-1]:
        fp.write('' + feat[2][0] + ', ')
    fp.write(feat_details[-1][2][0] + '};\n')

    fp.write(FX_CATALOG_C_LINES["ptrs"])
    for feat in feat_details[:-1]:
        fp.write('&fx_' + feat[0] + ', ')
    fp.write('&fx_' + feat_details[-1][0] + '};\n')

    fp.write(FX_CATALOG_C_LINES["width"])
    for feat in feat_details[:-1]:
        fp.write('' + feat[2][3] + ', ')
    fp.write(feat_details[-1][2][3] + '};\n')


        
FILE_HEADER = """
/* $Id:
 *=======================================================================
 *
 *                       Cereproc Ltd.
 *                       Copyright (c) 2006
 *                       All Rights Reserved.
 *
 *=======================================================================
 */

/* Automatically generated:"""

FX_CATALOG_H_HEAD = """

/* Feature extraction
 */

#if !(defined __FX_CATALOG_H__)
#define __FX_CATALOG_H__

#include <CPRC_features.h>

#define CPRC_NO_FEATURES"""


FX_CATALOG_H_TAIL = """

#endif /* __FX_CATALOG_H__ */
"""

FX_CATALOG_C_HEAD = """
#include <CPRC_features.h>
#include <CPRC_fx_catalog.h>

"""
FX_CATALOG_C_LINES = {"lbls":"const char * const CPRC_FX_LBL[] = {", \
                      "runbuild":"const enum CPRC_VOICE_BUILD CPRC_FX_BUILD[] = {", \
                      "type":"const enum CPRC_FX_TYPE CPRC_FX_TYPES[] = {", \
                      "ptrs":"const CPRC_fx_function CPRC_FX_FUNCTIONS[] = {", \
                      "width": "const int CPRC_FX_WIDTH[] = {"}


def get_dm_details(srcdir):
    files = glob.glob(os.path.join(srcdir, "dmetrics", "dm*.c"))
    files.sort()
    dm_details = []
    for f in files:
        print f
	fname = os.path.split(f)[1]
        pat = re.match("dm_(.*)\.c", fname)
        dmname = pat.group(1)
        if dmname == "catalog":
            continue
        prototype = ""
        fwk = ""
        lines = open(f).readlines()
        for l in lines:
            #match function heading
            pat = re.match("(cprc_float\s+dm_" + dmname + ".*)\s*{.*", l)
            if pat:
                prototype = pat.group(1)
            pat = re.match("/\*\s+CPRC_CF_FRAMEWORK\s+(.*)\*/.*", l)
            if pat:
                fwk = pat.group(1).split()
        dm_details.append([dmname, prototype, fwk, 0])
    return dm_details

#new dms (protected agianst reordering
def get_DM_details(srcdir):
    files = glob.glob(os.path.join(srcdir, "dmetrics", "DM*.c"))
    files.sort()
    dm_details = []
    for f in files:
        print f
	fname = os.path.split(f)[1]
        pat = re.match("DM_(.*)\.c", fname)
        dmname = pat.group(1)
        if dmname == "catalog":
            continue
        prototype = ""
        fwk = ""
        lines = open(f).readlines()
        for l in lines:
            #match function heading
            pat = re.match("(cprc_float\s+dm_" + dmname + ".*)\s*{.*", l)
            if pat:
                prototype = pat.group(1)
            pat = re.match("/\*\s+CPRC_CF_FRAMEWORK\s+(.*)\*/.*", l)
            if pat:
                fwk = pat.group(1).split()
        dm_details.append([dmname, prototype, fwk, 1])
    return dm_details

def generate_dm_cmake(srcdir, dm_details):
    fp = open(os.path.join(srcdir, "dmetrics", "dm.cmake"), 'w')
    fp.write("# Automatically generated: " + time.asctime() + "\n")
    fp.write("INCLUDE_DIRECTORIES(${MY_SOURCE_DIR}/dmetrics)\nSET(CEREVOICE_DM\n")
    fp.write("\t\tdmetrics/dm_catalog.c\n")
    for dm in dm_details:
        if dm[3]:
            fp.write("\t\tdmetrics/DM_" + dm[0] + ".c\n")
        else:
            fp.write("\t\tdmetrics/dm_" + dm[0] + ".c\n")      
    fp.write(")\n")
    fp.close()
    
def generate_dm_catalog_h(srcdir, dm_details):
    fp = open(os.path.join(srcdir, "dmetrics", "CPRC_dm_catalog.h"), 'w')
    fp.write(FILE_HEADER)
    fp.write(" " + time.asctime() + "*/\n")
    fp.write(DM_CATALOG_H_HEAD)
    fp.write(" " + str(len(dm_details)) + "\n\n")
    for dm in dm_details:
        fp.write(dm[1] + ";\n")
    fp.write(DM_CATALOG_H_TAIL)
    fp.close()
    
def generate_dm_catalog_c(srcdir, dm_details):
    fp = open(os.path.join(srcdir,"dmetrics", "dm_catalog.c"), 'w')
    fp.write(FILE_HEADER)
    fp.write(" " + time.asctime() + "*/\n")
    fp.write(DM_CATALOG_C_HEAD)

    fp.write(DM_CATALOG_C_LINES["lbls"])
    for dm in dm_details[:-1]:
        fp.write('"' + dm[0] + '", ')
    fp.write('"' + dm_details[-1][0] + '"};\n')

    fp.write(DM_CATALOG_C_LINES["no_feats"])
    for dm in dm_details[:-1]:
        fp.write(dm[2][1] + ', ')
    fp.write(dm_details[-1][2][1] + '};\n')

    fp.write(DM_CATALOG_C_LINES["feat_idx"])
    idx = 0
    for dm in dm_details[:-1]:
        fp.write(str(idx) + ', ')
        idx += int(dm[2][1])
    fp.write(str(idx) + '};\n')

    fp.write(DM_CATALOG_C_LINES["feat_widths"])
    for dm in dm_details[:-1]:
        widths = string.split(dm[2][3], "_")
        widths = string.join(widths, ", ")
        fp.write(widths + ', ')
    widths = string.split(dm_details[-1][2][3], "_")
    widths = string.join(widths, ", ")
    fp.write(widths + '};\n')

    fp.write(DM_CATALOG_C_LINES["no_params"])
    for dm in dm_details[:-1]:
        fp.write(dm[2][5] + ', ')
    fp.write(dm_details[-1][2][5] + '};\n')

    fp.write(DM_CATALOG_C_LINES["param_idx"])
    idx = 0
    for dm in dm_details[:-1]:
        fp.write(str(idx) + ', ')
        idx += int(dm[2][5])
    fp.write(str(idx) + '};\n')
    tot_no_params = idx + int(dm_details[-1][2][5])

    fp.write(DM_CATALOG_C_LINES["ptrs"])
    for dm in dm_details[:-1]:
        fp.write('&dm_' + dm[0] + ', ')
    fp.write('&dm_' + dm_details[-1][0] + '};\n')

    dm_param = []
    fp.write(DM_CATALOG_C_LINES["param_lbls"])
    for dm in dm_details:
        for i in range(0, int(dm[2][5]) * 3, 3):
            dm_param.append('"' + dm[2][6 + i] + '"')
    
    fp.write(", ".join(dm_param) + '};\n')

    dm_param = []
    fp.write(DM_CATALOG_C_LINES["param_feats"])
    for dm in dm_details:
        for i in range(2, int(dm[2][5]) * 3, 3):
            dm_param.append(dm[2][6 + i])
    
    fp.write(", ".join(dm_param) + '};\n')


    

    if tot_no_params == 0:
        fp.write('""};\n')

DM_CATALOG_H_HEAD = """

/* Distance Metrics
 */

#if !(defined __DM_CATALOG_H__)
#define __DM_CATALOG_H__

#include <CPRC_cf.h>

#define CPRC_NO_DMS"""

DM_CATALOG_H_TAIL = """

#endif /* __DM_CATALOG_H__ */
"""

DM_CATALOG_C_HEAD = """
#include <CPRC_cf.h>
#include <CPRC_dm_catalog.h>

"""

DM_CATALOG_C_LINES = {"lbls":"const char * const CPRC_DM_LBL[] = {", \
                      "no_feats": "const int CPRC_DM_NO_FEATS[] = {", \
                      "feat_idx": "const int CPRC_DM_FEAT_IDX[] = {",\
                      "feat_widths": "const int CPRC_DM_FEAT_WIDTHS[] = {",\
                      "no_params": "const int CPRC_DM_NO_PARAMS[] = {", \
                      "param_idx": "const int CPRC_DM_PARAM_IDX[] = {",\
                      "ptrs":"const CPRC_dm_function CPRC_DM_FUNCTIONS[] = {", \
                      "param_lbls":"const char * const CPRC_DM_PLBL[] = {", \
                      "param_feats":"const int CPRC_DM_PFEAT[] = {"}

def main(argv=None):
    #get arguments
    if not argv:
        argv = sys.argv

    if len(argv) < 2 or  argv[1] == '-h':
        print "python create_framework <source dir>"
        print "\te.g. python create_framework cereproc/cerevoice/src"
        sys.exit(1)
    srcdir = argv[1]
    feat_details = get_fx_details(srcdir)
    if (len(feat_details) == 0):
        print "Error: No feature code found\n"
        sys.exit(1)
    generate_fx_cmake(srcdir, feat_details) 
    generate_fx_catalog_h(srcdir, feat_details) 
    generate_fx_catalog_c(srcdir, feat_details) 

    #legacy dmetrics
    dm_details = get_dm_details(srcdir)
    #new dmetrics
    dm_details = dm_details + get_DM_details(srcdir)
    if (len(dm_details) == 0):
        print "Error: No distance metric code found\n"
        sys.exit(1)
    generate_dm_cmake(srcdir, dm_details) 
    generate_dm_catalog_h(srcdir, dm_details) 
    generate_dm_catalog_c(srcdir, dm_details) 
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
