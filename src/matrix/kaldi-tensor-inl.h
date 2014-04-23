// matrix/kaldi-tensor-inl.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_MATRIX_KALDI_TENSOR_INL_H_
#define KALDI_MATRIX_KALDI_TENSOR_INL_H_ 1

namespace kaldi {

template<typename Real>
Real& Tensor<Real>::operator() (int32 d0, int32 d1, int32 d2, int32 d3, int32 d4) {
  // might later turn this to KALDI_PARANOID_ASSERT if it takes too long.  
  KALDI_ASSERT(static_cast<size_t>(d0) < static_cast<size_t>(dims_[0].dim) &&
               static_cast<size_t>(d1) < static_cast<size_t>(dims_[1].dim) &&
               static_cast<size_t>(d2) < static_cast<size_t>(dims_[2].dim) &&
               static_cast<size_t>(d3) < static_cast<size_t>(dims_[3].dim) &&
               static_cast<size_t>(d4) < static_cast<size_t>(dims_[4].dim));
  return data_[d0 * dims_[0].stride + d1 * dims_[1].stride
               + d2 * dims_[2].stride + d3 * dims_[3].stride
               + d4 * dims_[4].stride];
}

template<typename Real>
Real Tensor<Real>::operator()(int32 d0, int32 d1, int32 d2, int32 d3, int32 d4) const {
  // might later turn this to KALDI_PARANOID_ASSERT if it takes too long.
  KALDI_ASSERT(static_cast<size_t>(d0) < static_cast<size_t>(dims_[0].dim) &&
               static_cast<size_t>(d1) < static_cast<size_t>(dims_[1].dim) &&
               static_cast<size_t>(d2) < static_cast<size_t>(dims_[2].dim) &&
               static_cast<size_t>(d3) < static_cast<size_t>(dims_[3].dim) &&
               static_cast<size_t>(d4) < static_cast<size_t>(dims_[4].dim));
  return data_[d0 * dims_[0].stride + d1 * dims_[1].stride
               + d2 * dims_[2].stride + d3 * dims_[3].stride
               + d4 * dims_[4].stride];
}



} // namespace kaldi


#endif  // KALDI_MATRIX_KALDI_TENSOR_INL_H_
