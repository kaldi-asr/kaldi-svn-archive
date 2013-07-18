// bin/fulctx-to-pdf.cc

// Copyright 2013  Arnab Ghoshal

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

// Converts frame-level full-context alignments to pdfs. The full-context model
// names are vectors of integers, each of whose dimension correspond to a feature
// type (e.g. phone identity, stress, syllable position, etc.). The last element
// is the pdf-class (HMM state, e.g. 0, 1, 2 for a 3-state HMM). This is used to
// query the decision tree to obtain the pdf id, which is similar to the function
// of the H tranducer in the ASR case. This is intended for use during synthesis.
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Converts frame-level full-context alignments to pdf-ids.\n"
        "Usage:  fullctx-to-pdf  [options] <tree> <alignments-rspecifier> <pdfs-wspecifier>\n"
        "e.g.: \n"
        " fullctx-to-pdf tree ark:1.ali ark:1.pdf.ali\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2),
        pdfs_wspecifier = po.GetArg(3);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    SequentialInt32VectorVectorReader ali_reader(alignments_rspecifier);
    Int32VectorWriter writer(pdfs_wspecifier);

    int32 num_done = 0;
    for (; !ali_reader.Done(); ali_reader.Next()) {
      std::string utt = ali_reader.Key();
      const std::vector< std::vector<int32> > &fullctx_ali(ali_reader.Value());
      std::vector<int32> pdf_ali(fullctx_ali.size());

      for (size_t i = 0; i < fullctx_ali.size(); i++) {
        std::vector<int32> fullctx(fullctx_ali[i].begin(), fullctx_ali[i].end()-1);
        int32 pdf_class = (*fullctx_ali[i].end());
        int32 pdf_id;
        if (!ctx_dep.Compute(fullctx, pdf_class, &pdf_id)) {
          std::ostringstream ctx_ss;
          WriteIntegerVector(ctx_ss, false, fullctx_ali[i]);
          KALDI_ERR << "Decision tree did not produce an answer for: pdf-class = "
                    << pdf_class << " context window = " << ctx_ss.str();
        }
        pdf_ali[i] = pdf_id;
      }

      writer.Write(utt, pdf_ali);
      num_done++;
    }
    KALDI_LOG << "Converted " << num_done << " alignments to pdf sequences.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


