#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-

#
# Copyright 2012 Lucas Ondel lucas.ondel@gmail.com
#

import argparse
import math

description = "Convert the KWS kaldi output to a MLF file."
postproc = ["none", "logadd", "bestscore"]

def overlap(start1, end1, start2, end2 ):
  if start2 < end1:
    return True
  return False

def average_timing(group):
  sum_start = 0
  sum_end = 0
  for (start, end, score) in group:
    sum_start += int(start)
    sum_end += int(end)
  return (sum_start / len(group), sum_end / len(group))
  
def compute_score(group):
  exponential_sum = 0
  for (start, end, score) in group:
    exponential_sum += math.exp(score)
  return  math.log(exponential_sum)
  
def main():
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("kws_results",                      
                      help="Results of the KWS system.")
  parser.add_argument("--postproc",
                      help="The kind of post processing which should be applied",
  					  dest="postproc",
  					  choices=postproc)
  args = parser.parse_args()
    
  results = open(args.kws_results).readlines()
  results.sort()
  
  processed_results = dict()
  for line in results:
    #Line is formatted as below:
    #KWID UTTID start_time end_time score    
    (kwid, utt, start, end, strscore) = line.split()
    
    # Invert score
    score = -1 * float(strscore)
    
    if (kwid, utt) not in processed_results:
      processed_results.setdefault((kwid, utt), [[(start, end, score)]])
    else:
      added = False
      for group in processed_results[(kwid, utt)]:
        (avg_start, avg_end) = average_timing(group)
        if overlap(avg_start, avg_end, int(start), int(end)):
          group.append((start, end, score))
          group.sort(key=lambda tup: tup[2])
          group.reverse()
          added = True      
      if not added:
        processed_results[(kwid, utt)].append([(start, end, score)])
  
  # Print the processed results.
  for (kwid, utt) in sorted(processed_results.keys()):
    for group in processed_results[(kwid, utt)]:
    #if len(processed_results[(kwid, utt)]) > 1:
    #  print("MULT KW ", kwid, " ", utt, " ", processed_results[(kwid, utt)]) 

      if args.postproc == postproc[0]: #none
        for (start, end, score) in group:
          print(kwid, " ", utt, " ", start, " ", end, " ", score, " ")
      if args.postproc == postproc[1]: #logadd
        (start, end, score) = group[0]
        print(kwid, " ", utt, " ", start, " ", end, " ", compute_score(group), " ")
      if args.postproc == postproc[2]: #bestscore
        (start, end, score) = group[0]
        print(kwid, " ", utt, " ", start, " ", end, " ", score, " ")
     

if __name__ == "__main__":
  main()