// bin/build-tree.cc

// Copyright 2013  Korbinian Riedhammer

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

#include "tree/topo-tree.h"
#include "tree/build-topo-tree.h"
#include "tree/clusterable-classes.h"
#include "tree/build-tree-utils.h"
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"

using namespace std;
using namespace kaldi;

int main(int argc, char **argv) {
  try {

    const char *usage =
      "Construct a topological tree.\n"
      "Usage:  build-topo-tree [options] <tree-stats-in> <tree-out>\n"
      "e.g.: \n"
      " build-tree treeacc tree\n";

    bool binary = true;
    int32 P = 1, N = 3;
    int32 max_leaves = 0;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("context-width", &N, "Context window size [must match "
                "acc-tree-stats]");
    po.Register("central-position", &P, "Central position in context window "
                "[must match acc-tree-stats]");
    po.Register("max-leaves", &max_leaves, "Maximum number of leaves to be "
                "used in tree-buliding (if positive)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string stats_filename = po.GetArg(1), tree_out_filename = po.GetArg(2);

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size();


    //////// Build the tree. ////////////
    TopoTree *tree = BuildTopoTree(stats, N, P);

    tree->Fill();

    tree->Print(std::cout);

    WriteKaldiObject(*tree, tree_out_filename, binary);

    KALDI_LOG<< "Wrote tree";

    DeleteBuildTreeStats(&stats);
  }
  catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

  return 0;
}
