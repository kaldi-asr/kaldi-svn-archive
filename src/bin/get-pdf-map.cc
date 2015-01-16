// bin/get-pdf-map.cc

// Copyright 2014 Xiaohui Zhang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

/** @brief Converts alignments (containing transition-ids) to pdf-ids, zero-based.
*/
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "hmm/posterior.h" 

namespace kaldi {
struct hash_pair {
  size_t operator() (const std::pair<int32, int32> &x ) const {
    return std::tr1::hash<int32>()(x.first) ^ std::tr1::hash<int32>()(x.second);  
  }
};
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Computes a soft pdf mapping from one system to another, for use in neural-net.\n"
        "training on soft labels derived from other neural-net systems.\n"
        "See also: apply-pdf-map. \n"
        "Usage:  get-pdf-map <src-alignments-rspecifier> <dest-alignments-rspecifier> <pdf-map-wxfilename> \n"
        "e.g.: \n"
        " get-pdf-map ark:1_src.ali ark:1_dest.ali pdf.map \n";
    ParseOptions po(usage);

    int32 src_num_pdfs;    
    int32 dest_num_pdfs;    
    po.Register("src_num_pdfs", &src_num_pdfs,
                "Number of pdfs/leaves in the source system.");
    po.Register("dest_num_pdfs", &dest_num_pdfs,
                "Number of pdfs/leaves in the destination system.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string src_alignments_rspecifier = po.GetArg(1),
        dest_alignments_rspecifier = po.GetArg(2),
        pdf_map_wxfilename = po.GetArg(3);

    SequentialInt32VectorReader src_reader(src_alignments_rspecifier);
    RandomAccessInt32VectorReader dest_reader(dest_alignments_rspecifier);

    std::tr1::unordered_map<std::pair<int32, int32>, int32, hash_pair> pair_counts;
    std::tr1::unordered_map<int32, int32> src_counts;
    Posterior pdf_map;

    int32 num_done = 0;
    int32 num_no_ali = 0;
    for (; !src_reader.Done(); src_reader.Next()) {
      std::string key = src_reader.Key();
      std::vector<int32> src_alignment = src_reader.Value();
      if (!dest_reader.HasKey(key)) {
        KALDI_WARN << "In destination alignments, did not find alignment for utterance " << key;
        num_no_ali++;
        continue;
      } else {
        std::vector<int32> dest_alignment = dest_reader.Value(key);
        for (size_t i = 0; i < src_alignment.size(); i++) {
          std::pair<int32, int32> pdf_pair = std::make_pair(src_alignment[i], dest_alignment[i]);
          if (pair_counts.find(pdf_pair) == pair_counts.end()) {
            pair_counts[pdf_pair] = 1;
          } else {
            pair_counts[pdf_pair] += 1;
          }
          if (src_counts.find(src_alignment[i]) == src_counts.end()) {
            src_counts[src_alignment[i]] = 1;
          } else {
            src_counts[src_alignment[i]] += 1;
          }  
        }
      }
      num_done++;
    }
    for (size_t i = 0; i < src_num_pdfs; i++) {
      std::vector<std::pair<int32, BaseFloat> > tmp;
      pdf_map.push_back(tmp);
      if (src_counts.find(i) == src_counts.end()) {
        KALDI_WARN << "In source alignments, did not find pdf  " << i << ". No mapping output for this pdf.";
      } else {
        for (size_t j = 0; j < dest_num_pdfs; j++) {
          std::pair<int32, int32> pdf_pair = std::make_pair(i, j);
          if (pair_counts.find(pdf_pair) != pair_counts.end()) {
            BaseFloat prob = static_cast<BaseFloat>(pair_counts[pdf_pair]) / static_cast<BaseFloat>(src_counts[i]);
            pdf_map[i].push_back(std::make_pair(j, prob)); 
          }
        }
        BaseFloat sum = 0.0;
        for (size_t j = 0; j < pdf_map[i].size(); j++ ) {
          sum += pdf_map[i][j].second;
        }
      }
    }  
    {
      Output ko(pdf_map_wxfilename, false);
      WritePosterior(ko.Stream(), false, pdf_map);
    }
    KALDI_LOG << "Generated a stochastic mapping using " << num_done << " alignments.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


