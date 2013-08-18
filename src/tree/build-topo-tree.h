// tree/build-topo-tree.h

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

#ifndef KALDI_TREE_BUILD_TOPO_TREE_H_
#define KALDI_TREE_BUILD_TOPO_TREE_H_

#include "tree/build-tree-utils.h"
#include "tree/topo-tree.h"

namespace kaldi {

TopoTree *BuildTopoTree(const BuildTreeStatsType &stats, int32 N, int32 P);

} // end namespace kaldi.



#endif /* KALDI_TREE_BUILD_TOPO_TREE_H_ */
