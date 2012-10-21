// nnet-cpubin/nnet-train-transitions.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet-cpu/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train the transition probabilities of a neural network acoustic model\n"
        "\n"
        "Usage:  nnet-train-transitions [options] <nnet-in> <alignments-rspecifier> <nnet-out>\n"
        "e.g.:\n"
        " nnet-train-transitions 1.nnet \"ark:gunzip -c ali.*.gz|\" 2.nnet\n";
    
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        ali_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);
    
    HmmTopology topo;
    ReadKaldiObject(topo_rxfilename, &topo);

    // Construct the transition model from the tree and the topology file.
    TransitionModel trans_model(ctx_dep, topo);

    AmNnet am_nnet;
    {
      bool binary;
      Input ki(config_rxfilename, &binary);
      KALDI_ASSERT(!binary && "Expect config file to contain text.");
      am_nnet.Init(ki.Stream());
    }

    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Initialized neural net and wrote it to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


