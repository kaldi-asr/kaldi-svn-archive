// bin/map-pdf-post.cc

// Copyright 2014  Mobvoi Inc. (Author: Xiaohui Zhang)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Map pdf-posteriors using a pdf-id to pdf-id mapping.\n"
        "the mapping is a vector<int32> and will generally correspond to deterministically mapping from \n"
        "a larger to a smaller tree, created by shrink-tree. \n" 
        "See also: apply-pdf-map which applys a stochastic mapping to pdf posteriors.\n"
        "Usage: map-pdf-post [options] <posteriors-rspecifier> <pdf-map> <posteriors-wspecifier>\n"
        "e.g.: \n"
        "map-pdf-post ark:1.post tree_map ark, t:-\n";
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1),
        pdf_map_rxfilename = po.GetArg(2),
        posteriors_wspecifier = po.GetArg(3);

    
    bool binary_in;
    std::vector<int32> pdf_map;
    Input ki(pdf_map_rxfilename, &binary_in);
    ReadIntegerVector(ki.Stream(), binary_in, &pdf_map);

    int32 num_done = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const kaldi::Posterior &posterior = posterior_reader.Value();
      kaldi::Posterior posterior_merged_vec;
      posterior_merged_vec.resize(posterior.size());
      for (int32 i = 0; i < posterior.size(); i++) {
        std::map<int32, BaseFloat> posterior_merged;
        for (int32 j = 0; j < posterior[i].size(); j++) {
          std::pair<std::map<int32,BaseFloat>::iterator,bool> ret;
          ret = posterior_merged.insert(std::pair<int32,BaseFloat>(pdf_map[posterior[i][j].first], posterior[i][j].second));
          if (ret.second == false) 
            posterior_merged[ret.first->first] += posterior[i][j].second;
        }
        for (std::map<int32,BaseFloat>::iterator iter = posterior_merged.begin();
             iter != posterior_merged.end(); ++iter) {
          posterior_merged_vec[i].push_back(*iter);
        }
      }
      posterior_writer.Write(posterior_reader.Key(), posterior_merged_vec);
      num_done++;
    }

    KALDI_LOG << "Merged " << num_done << " pdf soft alignments.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


