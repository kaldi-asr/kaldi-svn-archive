// nnet_dp/nnet1.h

// Copyright 2012  Daniel Povey

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

#ifndef KALDI_NNET_DP_AM_NNET1_H_
#define KALDI_NNET_DP_AM_NNET1_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

/*
  The class AmNnet1 has the job of taking the "Nnet1" class, which has a relatively
  simple interface, and giving it an interface that's suitable for acoustic modeling,
  dealing with 2-level trees, and so on.  Basically, this class handles various
  integer mappings that relate to the 2-level trees.
  
  Note: this class deals with setting up, and with storing, the neural net, but
  the likelihood computation is done by a separate class.
*/


class AmNnet1 {

  AmNnet1(const Nnet1InitConfig &config,
          std::vector<int32> leaf_mapping);
  // The vector "leaf_mapping" is an output from the program build-tree-two-level.
  // This maps from the leaves of the tree ("level-2 leaves") to the coarser "level-1"
  // leaves.  I.e. it's a fine-to-coarse mapping.
  
  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);

 private:
  Nnet1 nnet;
};



} // namespace

#endif // KALDI_NNET_DP_AM_NNET1_H_
