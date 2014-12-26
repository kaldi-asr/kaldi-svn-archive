// bin/shrink-tree.cc

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
#include "util/common-utils.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Shrink a decision tree (with a specified clustering threshold) and get the mapping between the \n"
        "leaves (pdf-ids) of the original big tree and the shrank small tree. \n"
        "See also: build-tree-two-level\n"
        "Usage:  shrink-tree [options] <tree-stats-in> <tree-in> <map-out> <tree-out>\n"
        "e.g.: \n"
        " shrink-tree treeacc roots.txt 1.qst topo tree_map tree_in tree_out\n";

    bool binary = true;
    int32 P = 1, N = 3;
    BaseFloat cluster_thresh = 0.1;  // negative means use smallest split in splitting phase as thresh.
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("context-width", &N, "Context window size [must match "
                "acc-tree-stats]");          
    po.Register("central-position", &P, "Central position in context window "
                "[must match acc-tree-stats]");
    po.Register("cluster-thresh", &cluster_thresh, "Log-likelihood change "
                "threshold for clustering after tree-building.  0 means "
                "no clustering; -1 means use as a clustering threshold the "
                "likelihood change of the final split.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_filename = po.GetArg(1),
        tree_in_filename = po.GetArg(2),
        map_out_filename = po.GetArg(3),
        tree_out_filename = po.GetArg(4);


    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size();
    
    ContextDependency ctx_dep_in;
    ReadKaldiObject(tree_in_filename, &ctx_dep_in);
    
    std::vector<EventMap*> mapping;
    int32 num_reduced = ClusterEventMapGetMapping(ctx_dep_in.ToPdfMap(), stats, cluster_thresh, &mapping);
    std::vector<int32> leaves_mapping;

    EventMap *clustered_map = ctx_dep_in.ToPdfMap().Copy(mapping);
    EventAnswerType new_nleaves;
    EventMap *renumbered_map = RenumberEventMap(*clustered_map, &new_nleaves);

    ComputeTreeMapping(*renumbered_map, ctx_dep_in.ToPdfMap(), stats, &leaves_mapping);

    {
      Output ko(map_out_filename, binary);
      WriteIntegerVector(ko.Stream(), binary, leaves_mapping);
    }

    { // This block is to warn about low counts.
      std::vector<BuildTreeStatsType> split_stats;
      SplitStatsByMap(stats, *renumbered_map,
                      &split_stats);
      for (size_t i = 0; i < split_stats.size(); i++)
        if (SumNormalizer(split_stats[i]) < 100.0)
          KALDI_VLOG(1) << "For pdf-id " << i << ", low count "
                        << SumNormalizer(split_stats[i]);
    }
    
    ContextDependency ctx_dep(N, P, renumbered_map);  // takes ownership
    // of pointer "to_pdf", so set it NULL.
    renumbered_map = NULL;

    WriteKaldiObject(ctx_dep, tree_out_filename, binary);
    KALDI_LOG << "Shrinking finished. Reduced " << num_reduced << " leaves." ;
    KALDI_LOG << "Wrote tree";

    DeleteBuildTreeStats(&stats);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
