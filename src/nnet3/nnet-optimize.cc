// nnet3/nnet-optimize.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace nnet3 {


void IdentifySubmatrixArgs(NnetComputation::Command *c,
                           std::vector<int32*> *submatrix_args) {
  submatrix_args->clear();
  switch (c->command_type) {
      case NnetComputation::kResizeMatrixZeroed:
      case NnetComputation::kResizeMatrixUndefined:
      case NnetComputation::kResizeMatrixEmpty:
        break;
    case NnetComputation::kPropagate:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
      break;
    case NnetComputation::kStoreStats:
      submatrix_args->push_back(&c->arg2);
      break;
    case NnetComputation::kBackprop:
      submatrix_args->push_back(&c->arg4);
      submatrix_args->push_back(&c->arg5);
      submatrix_args->push_back(&c->arg6);
      submatrix_args->push_back(&c->arg7);      
      break;
    case NnetComputation::kMatrixCopy:
    case NnetComputation::kMatrixAdd:
    case NnetComputation::kAddRows:
    case NnetComputation::kCopyRows:
    case NnetComputation::kAddRowRanges:
      submatrix_args->push_back(&c->arg1);
      submatrix_args->push_back(&c->arg2);
      break;
    case NnetComputation::kAddRowsMulti:
    case NnetComputation::kCopyRowsMulti:
    case NnetComputation::kAddToRowsMulti:
    case NnetComputation::kCopyToRowsMulti:
      submatrix_args->push_back(&c->arg1);
      break;
    case NnetComputation::kNoOperation:
    case NnetComputation::kNoOperationMarker:
      break;
    default:
      KALDI_ERR << "Unknown command type.";
  }
}
    
void IdentifyMatrixArgs(NnetComputation::Command *c,
                        std::vector<int32*> *matrix_args) {
  matrix_args->clear();
  switch (c->command_type) {
    case NnetComputation::kResizeMatrixZeroed:
    case NnetComputation::kResizeMatrixUndefined:
    case NnetComputation::kResizeMatrixEmpty:
      matrix_args->push_back(&c->arg1);
      break;
    default:
      break;
  }
}



} // namespace nnet3
} // namespace kaldi
