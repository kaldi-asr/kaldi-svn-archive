// gmmbin/gmm-train-trans.cc

// Copyright 2014 Mobvoi Inc. (Author: Xiaohui Zhang)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "tree/build-tree.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Train a trainsition model given topo, tree, alignments and GMM\n"
        "See also: gmm-init-trans, gmm-init-model-flat \n"
        "Usage:  gmm-train-trans <topology-in> <gmm-in> <tree-in> <ali-in> <model-out>\n";

    bool binary = true;
    MleTransitionUpdateConfig transition_update_config;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    transition_update_config.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_filename = po.GetArg(1);
    std::string gmm_filename = po.GetArg(2);
    std::string tree_filename = po.GetArg(3);
    std::string ali_rspecifier = po.GetArg(4);
    std::string model_out_filename = po.GetArg(5);
    // Read toppology.
    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    // Read model.
    bool binary_read;
    TransitionModel old_trans_model;
    AmDiagGmm am_gmm;
    Input ki(gmm_filename, &binary_read);
    old_trans_model.Read(ki.Stream(), binary_read);
    am_gmm.Read(ki.Stream(), binary_read);

    // Now the tree
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    TransitionModel trans_model(ctx_dep, topo);
    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);

    int32 num_done = 0;
    SequentialInt32VectorReader ali_reader(ali_rspecifier);
    for (; ! ali_reader.Done(); ali_reader.Next()) {
      const std::vector<int32> alignment(ali_reader.Value());
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i];
        BaseFloat weight = 1.0;
        trans_model.Accumulate(weight, tid, &transition_accs);
      }
      num_done++;
    }
    KALDI_LOG << "Accumulated transition stats from " << num_done
              << " utterances.";

    {
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, transition_update_config,
                            &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << count
                << " frames.";
    }

    {  // Write transition-model and GMM to one file in the normal Kaldi way.
      Output out(model_out_filename, binary);
      trans_model.Write(out.Stream(), binary);
      am_gmm.Write(out.Stream(), binary);
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

