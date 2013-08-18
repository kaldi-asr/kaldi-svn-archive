// tree/build-topo-tree.cc

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

#include "tree/build-topo-tree.h"
#include "tree/topo-tree.h"
#include "tree/build-tree-utils.h"

#include <utility>

namespace kaldi {

TopoTree *BuildTopoTree(const BuildTreeStatsType &stats, int32 N, int32 P) {
  KALDI_ASSERT(N > 0);
  KALDI_ASSERT(P > 0);
  KALDI_ASSERT(P < N);

  // our new tree object.
  TopoTree *tree = new TopoTree(N, P);

  // iterate over all stats, insert into new tree
  for (BuildTreeStatsType::const_iterator it = stats.begin(); it != stats.end(); it++) {

    EventType event_type((*it).first);

    if (IsContextIndependentEventType(event_type))
      event_type = PadCtxIndependentEventType(event_type, N, P);

    tree->Insert(event_type);
  }

  return tree;
}

} // end namespace kaldi.
