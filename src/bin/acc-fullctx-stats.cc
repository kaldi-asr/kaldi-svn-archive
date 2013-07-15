// bin/acc-fullctx-stats.cc

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

#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "tree/build-tree-utils.h"
#include "hmm/transition-model.h"
#include "hmm/tree-accu.h"

// This is similar to acc-tree-stats.cc, except for the following differences:
// 1. The 'alignments' are in terms of fully expanded model names that are
//    represented as std::vector<int32>, where each dimension is a feature type,
//    e.g. phone identity, stress, syllable position, etc. The last element is
//    for the pdf-class (HMM state, e.g. 0, 1, 2 for 3-state HMM).
// 2. No transition model is used as it is assumed that the full-context names
//    are in terms of phone identities, and not transition-ids.
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate statistics for phonetic-context tree building.\n"
        "Uses alignments with fully-expanded model names. Intended for "
        "parametric synthesis.\n\n"
        "Usage:  acc-fullctx-stats [options] central-position features-rspecifier"
        " alignments-rspecifier tree-accs-out\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat var_floor = 0.01;
    string ci_phones_str;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("ci-phones", &ci_phones_str, "Colon-separated list of integer "
                "indices of context-independent phones.");
    po.Register("var-floor", &var_floor, "Variance floor for tree clustering.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string central_pos_str = po.GetArg(1),
        feat_rspecifier = po.GetArg(2),
        ali_rspecifier = po.GetArg(3),
        accs_out_filename = po.GetArg(4);

    int32 central_pos;
    if (!ConvertStringToInteger(central_pos_str, &central_pos) || central_pos < 0)
      KALDI_ERR << "Invalid central position: expecting integer >= 0, found: "
                << central_pos_str;

    std::vector<int32> ci_phones;
    if (ci_phones_str != "") {
      SplitStringToIntegers(ci_phones_str, ":", false, &ci_phones);
      std::sort(ci_phones.begin(), ci_phones.end());
      if (!IsSortedAndUniq(ci_phones) || ci_phones[0] == 0) {
        KALDI_ERR << "Invalid set of ci_phones: " << ci_phones_str;
      }
    }

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessInt32VectorVectorReader ali_reader(ali_rspecifier);

    std::map<EventType, GaussClusterable*> tree_stats;

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      if (!ali_reader.HasKey(utt)) {
        num_no_alignment++;
        continue;
      }

      const Matrix<BaseFloat> &feats = feat_reader.Value();
      const vector< vector<int32> > &alignment = ali_reader.Value(utt);

      if (alignment.size() != feats.NumRows()) {
        KALDI_WARN << "Alignments has wrong size: " << alignment.size()
                   << " vs. " << feats.NumRows();
        num_other_error++;
        continue;
      }

      AccumulateFullCtxStats(alignment, feats, ci_phones, central_pos,
                             var_floor, &tree_stats);
      num_done++;
      if (num_done % 1000 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances.";
    }

    // Store in the vectorized form
    BuildTreeStatsType stats;
    for (std::map<EventType, GaussClusterable*>::const_iterator iter =
         tree_stats.begin(); iter != tree_stats.end(); ++iter) {
      stats.push_back(std::make_pair<EventType, GaussClusterable*>(iter->first,
                                                                   iter->second));
    }
    tree_stats.clear();

    {
      Output ko(accs_out_filename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }

    KALDI_LOG << "Accumulated stats for " << num_done << " files, "
              << num_no_alignment << " failed due to no alignment, "
              << num_other_error << " failed for other reasons.";
    KALDI_LOG << "Number of separate stats (context-dependent states) is "
              << stats.size();
    DeleteBuildTreeStats(&stats);

    if (num_done != 0)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
