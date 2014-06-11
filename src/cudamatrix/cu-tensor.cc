// cudamatrix/cu-tensor.cc

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

#include <algorithm>
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-tensor.h"
#include "matrix/kaldi-tensor-internals.h" // for TensorOperationDims

namespace kaldi {


template<typename Real>
CuTensor<Real>::CuTensor(const CuMatrixBase<Real> &mat,
                         int32 dim0, int32 stride0,
                         int32 dim1, int32 stride1,
                         int32 dim2, int32 stride2) {
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(3);
  dims_strides.push_back(std::make_pair(dim0, stride0));
  dims_strides.push_back(std::make_pair(dim1, stride1));
  dims_strides.push_back(std::make_pair(dim2, stride2));
  Init(dims_strides, mat);
}


template<typename Real>
CuTensor<Real>::CuTensor(const CuMatrixBase<Real> &mat,
                       int32 dim0, int32 stride0,
                       int32 dim1, int32 stride1,
                       int32 dim2, int32 stride2,
                       int32 dim3, int32 stride3) {
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(4);
  dims_strides.push_back(std::make_pair(dim0, stride0));
  dims_strides.push_back(std::make_pair(dim1, stride1));
  dims_strides.push_back(std::make_pair(dim2, stride2));
  dims_strides.push_back(std::make_pair(dim3, stride3));
  Init(dims_strides, mat);
}

template<typename Real>
CuTensor<Real>::CuTensor(const CuMatrixBase<Real> &mat,
                       int32 dim0, int32 stride0,
                       int32 dim1, int32 stride1,
                       int32 dim2, int32 stride2,
                       int32 dim3, int32 stride3,
                       int32 dim4, int32 stride4) {
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(5);
  dims_strides.push_back(std::make_pair(dim0, stride0));
  dims_strides.push_back(std::make_pair(dim1, stride1));
  dims_strides.push_back(std::make_pair(dim2, stride2));
  dims_strides.push_back(std::make_pair(dim3, stride3));
  dims_strides.push_back(std::make_pair(dim4, stride4));
  Init(dims_strides, mat);
}

template<typename Real>
CuTensor<Real>::CuTensor(const std::vector<std::pair<int32, int32> > &dims_strides,
                       const CuMatrixBase<Real> &mat) {
  Init(dims_strides, mat);
}

template<typename Real>
CuTensor<Real>::CuTensor(const std::vector<std::pair<int32, int32> > &dims_strides,
                       const Real *data) {
  Init(dims_strides, data);
}

template<typename Real>
void CuTensor<Real>::Init(const std::vector<std::pair<int32, int32> > &dims_strides,
                          const Real *data) {
  this->dims_strides_ = dims_strides;
  // Like CuSubMatrix, CuTensor is not const-correct, as it would be too hard to
  // keep track of whether a tensor had been initialized with a const reference
  // or a pointer.
  this->data_ = const_cast<Real*>(data);
  this->CheckAndFixDims();
}

template<typename Real>
void CuTensor<Real>::Init(const std::vector<std::pair<int32, int32> > &dims_strides,
                          const CuMatrixBase<Real> &mat) {
  this->Init(dims_strides, mat.Data());

  int32 min_offset, max_offset;
  this->GetMinAndMaxOffset(-1, &min_offset, &max_offset);
  KALDI_ASSERT(min_offset == 0 &&
               max_offset <=
               (mat.NumRows() - 1) * mat.Stride() + mat.NumCols() - 1);

  // Get min and max offset including only strides less than the matrix's
  // row stride.  The min and max offset should fit within the row.
  this->GetMinAndMaxOffset(mat.Stride(), &min_offset, &max_offset);
  KALDI_ASSERT(min_offset == 0 &&
               max_offset < mat.NumCols());
}

template<class Real>
void CuTensor<Real>::AddTensorTensor(Real alpha,
                                     const CuTensor<Real> &t1,
                                     const CuTensor<Real> &t2,
                                     Real beta) {
  KALDI_ASSERT(this->NumIndexes() == t1.NumIndexes() &&
               t1.NumIndexes() == t2.NumIndexes());
  
  // Dimension check.  For each index 0 <= i < NumIndexes(), let
  // a = this->Dim(i), b = t1.Dim(i) and c = t2.Dim(i).
  // Let m = max(a, b, c).
  // We require that each of a, b, c be equal to 1 or to m.
  // The simplest case is a = b = c = m.
  // We also allow things like a = b, c = 1 or a = 1, b = c,
  // which frequently arises; or even for only one of the
  // three to be nonzero, which is a little less common but
  // still valid.
  
  size_t order = this->dims_strides_.size();
  std::vector<TensorOperationDims> dims;
  dims.reserve(order);
  
  for (size_t i = 0; i < order; i++) {
    int32 a = this->dims_strides_[i].first,
        b = t1.dims_strides_[i].first,
        c = t2.dims_strides_[i].first,
        m = std::max(a, std::max(b, c));
    if (!(a == 1 || a == m) || !(b == 1 || b == m) || !(c == 1 || c == m)) {
      KALDI_ERR << "Tensor dimension mismatch: for index i = " << i
                << " this->Dim(i) = " << a << ", t1.Dim(i) = " << b
                << ", t2.Dim(i) = " << c;
    }
    dims.push_back(TensorOperationDims(m,
                                       this->dims_strides_[i].second,
                                       t1.dims_strides_[i].second,
                                       t2.dims_strides_[i].second));
  }
  // TODO.
  // AddTensorTensorToplevel(dims, alpha, t1.Data(), t2.Data(), this->Data(),
  //  beta);
}

template<class Real>
void CuTensor<Real>::AddTensor(Real alpha,
                               const CuTensor<Real> &t) {
  
  KALDI_ASSERT(this->NumIndexes() == t.NumIndexes());

  // Dimension check.  For each index 0 <= i < NumIndexes(), let
  // a = this->Dim(i), b = t1.Dim(i)
  // We must have either a == b or a = 1 or b = 1.
  
  size_t order = this->dims_strides_.size();
  std::vector<TensorOperationDims> dims;
  dims.reserve(order);
  
  for (size_t i = 0; i < order; i++) {
    int32 a = this->dims_strides_[i].first,
        b = t.dims_strides_[i].first,
        m = std::max(a, b);

    if (!(a == b || a == 1 || b == 1)) {
      KALDI_ERR << "Tensor dimension mismatch: for index i = " << i
                << " this->Dim(i) = " << a << ", t.Dim(i) = " << b;
    }
    // we set stride_a and stride_c; stride_b is always zero.
    dims.push_back(TensorOperationDims(m,
                                       this->dims_strides_[i].second,
                                       0,
                                       t.dims_strides_[i].second));
  }
  // TODO.
  // AddTensorToplevel(dims, alpha, t.Data(), this->Data());
}

template<typename Real>
void CuTensor<Real>::Scale(Real alpha) {
  CuTensor<Real> t(*this);  // Note: this uses the default copy constructor.
  t.Flatten();  // This removes aliasing.  It may fail if this is not possible
                // to do; this will only be the case for quite strange tensors,
                // and for now we just don't support scaling of such tensors.

  // TODO.
  //ScaleTensor(this->NumIndexes(),
  //              t.dims_strides_.empty() ? NULL : &(t.dims_strides_[0]),
  //              alpha, t.data_);
}

template<typename Real>
void CuTensor<Real>::CopyFromTensor(const CuTensor<Real> &t) {
  // TODO.
}

template<typename Real>
void CuTensor<Real>::ConvTensorTensor(Real alpha,
                                      const Tensor<Real> &t1,
                                      const Tensor<Real> &t2) {
  // TODO.
}

template<typename Real>
void CuTensor<Real>::ConvTensorTensor(Real alpha,
                                      const CuTensor<Real> &t1,
                                      const CuTensor<Real> &t2) {
  // TODO.
}

// Explicit instantiation of the classes
template class CuTensor<float>;
template class CuTensor<double>;

} // namespace kaldi

