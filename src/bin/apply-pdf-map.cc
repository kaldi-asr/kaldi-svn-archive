// bin/apply-pdf-map.cc

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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Apply a stochastic mapping to pdf posteriors, to get pdf posteriors for the destination system.\n"
        "The mapping is usually created by get-pdf-map. \n"
        "See also: map-pdf-post which applys a deterministic mapping to pdf posteriors.\n"
        "Usage:  apply-pdf-map <pdf-map-rxfilename> <src-alignments-rspecifier> <dest-alignments-rspecifier>  \n"
        "e.g.: \n"
        " apply-pdf-map pdf_map ark:1_src.ali ark:1_dest.ali \n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string pdf_map_rxfilename = po.GetArg(1),
        posteriors_rspecifier = po.GetArg(2),
        posteriors_wspecifier = po.GetArg(3);

    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    Posterior pdf_map;
    {
      bool binary;
      Input ki(pdf_map_rxfilename, &binary);
      ReadPosterior(ki.Stream(), &pdf_map);
    }

    int32 num_done = 0;
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const kaldi::Posterior &src_posterior = posterior_reader.Value();
      kaldi::Posterior dest_posterior;
      ApplyPdfMap(pdf_map, src_posterior, &dest_posterior);
      posterior_writer.Write(posterior_reader.Key(), dest_posterior);
      num_done++;
    }
    KALDI_LOG << "Done converting posteriors for " << num_done << " utterances.";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


