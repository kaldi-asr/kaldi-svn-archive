// nnet-dpbin/nnet1-init.cc

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
#include "nnet-dp/am-nnet1.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize neural network-based acoustic model (of type nnet1).\n"
        "Takes tree and mapping file as produced by build-tree-two-level\n"
        "Usage:  nnet1-init [options] <tree-in> <mapping-file> <topo-file> <nnet1-out>\n"
        "e.g.:\n"
        " nnet1-init tree tree.map topo 1.nnet1\n";
        
    bool binary_write = true;
    Nnet1InitConfig config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    config.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        mapping_rxfilename = po.GetArg(2),
        topo_rxfilename = po.GetArg(3),
        nnet1_wxfilename = po.GetArg(4);
    
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    std::vector<int32> mapping;

    {
      bool binary_read;
      Input ki(mapping_rxfilename, &binary_read);
      ReadIntegerVector(ki.Stream(), binary_read, &mapping); 
    }

    HmmTopology topo;
    ReadKaldiObject(topo_rxfilename, &topo);
    

    TransitionModel trans_model(ctx_dep, topo);
    AmNnet1 am_nnet1(config, mapping);
    
    {
      Output ko(nnet1_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet1.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Wrote model to " << nnet1_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


