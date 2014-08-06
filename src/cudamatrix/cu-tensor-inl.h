// matrix/cu-tensor-inl.h

// Copyright 2014  Pegah Ghahremani

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

#ifndef KALDI_CUDAMATRIX_CU_TENSOR_INL_H_
#define KALDI_CUDAMATRIX_CU_TENSOR_INL_H_ 1

namespace kaldi {

template<typename Real>
inline CuValue<Real> CuTensor<Real>::operator() (const std::vector<int32> &indexes) {
  KALDI_ASSERT(indexes.size() == this->dims_strides_.size());
  Real *data = this->data_;
  for (size_t i = 0; i < indexes.size(); i++) {
    size_t n = indexes[i],
        dim = this->dims_strides_[i].first,
        stride = this->dims_strides_[i].second;
    KALDI_ASSERT(n < dim);
    data += n * stride;
  }
  return CuValue<Real>(data);
}

template<typename Real>
inline Real CuTensor<Real>::operator() (const std::vector<int32> &indexes) const {
  KALDI_ASSERT(indexes.size() == this->dims_strides_.size());
  Real *data = this->data_;
  for (size_t i = 0; i < indexes.size(); i++) {
    size_t n = indexes[i],
        dim = this->dims_strides_[i].first,
        stride = this->dims_strides_[i].second;
    KALDI_ASSERT(n < dim);
    data += n * stride;
  }
  return CuValue<Real>(data);
}



} // namespace kaldi


#endif  // KALDI_MATRIX_KALDI_TENSOR_INL_H_
