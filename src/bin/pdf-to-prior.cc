// bin/pdf-to-prior.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include <numeric>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Reads int32 vectors representing pdf-ids, e.g. output by ali-to-pdf), and outputs\n"
        "normalized counts (priors) for each index, as a Vector<float>.\n"
        "\n"
        "Usage:  pdf-to-prior [options] <pdfs-rspecifier> <prior-wxfilname>\n"
        "e.g.: \n"
        " ali-to-pdf final.mdl \"ark:gunzip -c 1.ali.gz|\" ark:- | pdf-to-prior --num_pdfs=1946 ark:- prior.vec\n";
    ParseOptions po(usage);
    
    bool binary_write = false;
    bool log  = true;
    int num_pdfs = 0;
    BaseFloat prior_floor=1.0e-07;
    po.Register("num_pdfs", &num_pdfs, "We use num_pdfs to normalize the counts.");
    po.Register("binary", &binary_write, "Write in binary mode.");
    po.Register("log", &log, "Take log of the priors.");
    po.Register("prior_floor", &prior_floor, "The minimum value allowed for priors.");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(num_pdfs > 0 && "You must specify a positive integer num_pdfs.");
    std::string pdfs_rspecifier = po.GetArg(1),
        prior_wxfilename = po.GetArg(2);
    
    SequentialInt32VectorReader pdfs_reader(pdfs_rspecifier);

    std::vector<int64> counts(num_pdfs); // will turn to Vector<BaseFloat> after counting.
    int32 num_done = 0;
    for (; !pdfs_reader.Done(); pdfs_reader.Next()) {
      std::vector<int32> alignment = pdfs_reader.Value();
      
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 value = alignment[i];
        KALDI_ASSERT(value < counts.size() && "pdf_id exceeds num_pdfs.");
        counts[value]++; // accumulate counts
      }
      num_done++;
    }

    //convert to BaseFloat and write.
    Vector<BaseFloat> prior(counts.size());
    BaseFloat counts_sum = static_cast<BaseFloat>(std::accumulate(counts.begin(), counts.end(), 0));
    for(int32 i = 0; i < counts.size(); i++) {
      prior(i) = static_cast<BaseFloat>(counts[i]) / counts_sum;
      prior(i) = (prior(i) > prior_floor ? prior(i) : prior_floor);
      if (log) prior(i) = std::log(prior(i));
    }

    Output ko(prior_wxfilename, binary_write);
    prior.Write(ko.Stream(), binary_write);
    
    KALDI_LOG << "Took " << num_done << " int32 vectors to compute prior, "
              << "Number of pdfs is " << num_pdfs << ", total counts are "
              << counts_sum;
    return (num_done == 0 ? 1 : 0); // error exit status if processed nothing.
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


