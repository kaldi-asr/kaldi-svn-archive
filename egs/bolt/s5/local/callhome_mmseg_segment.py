#coding:utf-8
#!/usr/bin/env python
import sys
from mmseg import seg_txt
import re

nonspeech_events=re.compile(r"\<.*\>");
nonchinese_words=re.compile(r"[A-Z'-]+");

for line in sys.stdin:
  blks = str.split(line)
  out_line = blks[0]
  for i in range(1, len(blks)):
    if nonspeech_events.match(blks[i]) :
      out_line += " " + blks[i]
    elif  nonchinese_words.match(blks[i]):
      out_line += " " + blks[i]
    else:
      for j in seg_txt(blks[i]):
        out_line += " " + j
  print out_line     
