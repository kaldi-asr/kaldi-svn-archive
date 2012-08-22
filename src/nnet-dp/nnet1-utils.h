// nnet-dp/nnet1.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)


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

#ifndef KALDI_NNET_DP_NNET1_UTILS_H_
#define KALDI_NNET_DP_NNET1_UTILS_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

// This header contains certain functions used in the command-line tools,
// while training neural nets.

namespace kaldi {

void ReadAlignmentsAndFeatures(std::string feature_rspecifier,
                               std::string alignments_rspecifier,
                               std::string validation_utt_list,
                               std::vector<CompressedMatrix> *train_feats,
                               std::vector<CompressedMatrix> *validation_feats,
                               std::vector<std::vector<int32> > *train_ali,
                               std::vector<std::vector<int32> > *validation_ali) {
  
  

} // namespace

#endif
