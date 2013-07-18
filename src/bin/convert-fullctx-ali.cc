// bin/convert-fullctx-ali.cc

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
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"

// Converts an alignment (in terms of transition ids) produced by an initial
// model (which could be monophone or context-dependent); the corresponding
// full-context alignment (this is expected to include other features in
// addition to phonetic context, and hence cannot be deduced from the previous
// alignment alone), which is in terms of phone identities and not transition
// ids; and the full-context decision tree and models and produces alignments
// (in terms of transition ids) for the new model/tree. This is intended to be
// used during acoustic model training for synthesis.
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  using std::vector;
  try {
    const char *usage =
        "Convert alignments from one decision-tree/model to another\n"
        "Usage:  convert-fullctx-ali  [options] old-model new-model new-tree "
        "tid-ali-rspecifier fullctx-ali-respecifier new-tid-ali-wspecifier\n"
        "e.g.: \n"
        " convert-fullctx-ali old.mdl new.mdl new.tree ark:old.ali ark:full.ali"
        " ark:new.ali\n";


    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string old_model_filename = po.GetArg(1),
        new_model_filename = po.GetArg(2),
        new_tree_filename = po.GetArg(3),
        tid_ali_rspecifier = po.GetArg(4),
        fullctx_ali_rspecifier = po.GetArg(5),
        new_tid_ali_wspecifier = po.GetArg(6);

    TransitionModel old_trans_model;
    ReadKaldiObject(old_model_filename, &old_trans_model);
    TransitionModel new_trans_model;
    ReadKaldiObject(new_model_filename, &new_trans_model);

    if (!(old_trans_model.GetTopo() == new_trans_model.GetTopo()))
      KALDI_WARN << "Toplogies of models are not equal: "
                 << "conversion may not be correct or may fail.";

    ContextDependency new_ctx_dep;  // the tree.
    ReadKaldiObject(new_tree_filename, &new_ctx_dep);

    SequentialInt32VectorReader tid_ali_reader(tid_ali_rspecifier);
    RandomAccessInt32VectorVectorReader full_ali_reader(fullctx_ali_rspecifier);
    Int32VectorWriter alignment_writer(new_tid_ali_wspecifier);

    int32 num_success = 0, num_fail = 0;
    for (; !tid_ali_reader.Done(); tid_ali_reader.Next()) {
      std::string utt = tid_ali_reader.Key();
      if (!full_ali_reader.HasKey(utt)) {
        KALDI_WARN << "No full-context alignment found for utterance: " << utt;
        num_fail++;
        continue;
      }
      const vector<int32> &tid_ali(tid_ali_reader.Value());
      const vector< vector<int32> > &full_ali(full_ali_reader.Value(utt));
      vector<int32> new_alignment;
      if (ConvertFullCtxAlignment(old_trans_model, new_trans_model, new_ctx_dep,
                                  tid_ali, full_ali, &new_alignment)) {
        alignment_writer.Write(utt, new_alignment);
        num_success++;
      } else {
        KALDI_WARN << "Could not convert alignment for utterance: " << utt;
        num_fail++;
      }
    }

    KALDI_LOG << "Succeeded converting alignments for " << num_success
              << " files, failed for " << num_fail;
    if (num_success != 0) return 0;  //NOLINT
    else return 1;  //NOLINT
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
