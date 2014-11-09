#!/usr/bin/env python
# Converts a Kaldi text file (with segment information) to an SGM
# This can be used for 1-best output files and can be extended to n-best
# Expected line format : ar_4264-A-008853-009029 barDu istigmatizm [hes]
# Speaker and channel information is compressed into one entry here, 
#   may be useful if channel info is present especially when there are 
#   multiple speakers on the same channel. Let me know if this is the case
#   and I'll modify this script
# Assumes that the text file is sorted by document

import codecs
import sys

# Change if necessary
SYSID = "BOLT_CMN_CTS_JHU"

if len(sys.argv) < 3:
  print "Usage " + sys.argv[0] + " <kaldi-text-file> <output-xml>"
  sys.exit(1)


inputFile = codecs.open(sys.argv[1], encoding = "utf-8") or die ("Could not open text file for reading")
outputFile = codecs.open(sys.argv[2], "w+", encoding = "utf-8") or die ("Could not open file for writing")

currentDocument = None

for line in inputFile:
  line = line.strip()
  lineComp = line.split(" ")
  segID = lineComp[0]
  del lineComp[0]
  segData = " ".join(lineComp)
  segIDComp = segID.split("-")
  if len(segIDComp) != 4:
    print "Invalid segment ID at line : " + line
    sys.exit(1)

  documentID = segIDComp[0]
  channelID = segIDComp[1]
  # Note that the start and the end time format assume 6 digits, change if necessary
  startTime = str(float(segIDComp[2]) / 100);
  endTime = str(float(segIDComp[3]) / 100);

  if documentID != currentDocument:
    if currentDocument != None:
      outputFile.write("</DOC>\n")

    currentDocument = documentID
    uttID = 1
    outputFile.write("<DOC docid=\"" + currentDocument + "\" sysid=\"" + SYSID + "\">\n")

  outputFile.write("<seg id=\"" + str(uttID) + "\" speaker=\"" + channelID + "\" channel=\"" + ("1" if "B" in channelID else "0") + "\" begin=\"" + startTime + "\" end=\"" + endTime + "\"> " + segData+ " </seg>\n")
  uttID = uttID + 1

outputFile.write("</DOC>")

inputFile.close()
outputFile.close()
