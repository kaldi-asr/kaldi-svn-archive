// matrix/kaldi-tensor.cc

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
#include "matrix/kaldi-tensor.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

template<typename Real>
int32 Tensor<Real>::Dim(int32 index) const {
  KALDI_ASSERT(static_cast<uint32>(index) < 5);
  return dims_[index].dim;
}

template<typename Real>
int32 Tensor<Real>::Stride(int32 index) const {
  KALDI_ASSERT(static_cast<uint32>(index) < 5);
  return dims_[index].stride;
}


template<typename Real>
Tensor<Real>::Tensor(const MatrixBase<Real> &mat,
                     int32 dim1, int32 stride1,
                     int32 dim2, int32 stride2,
                     int32 dim3, int32 stride3,
                     int32 dim4, int32 stride4,
                     int32 dim5, int32 stride5):
    data_(const_cast<Real*>(mat.Data())) {
  dims_[0].dim = dim1;
  dims_[0].stride = stride1;
  dims_[1].dim = dim1;
  dims_[1].stride = stride1;
  dims_[2].dim = dim2;
  dims_[2].stride = stride2;
  dims_[3].dim = dim3;
  dims_[3].stride = stride3;
  dims_[4].dim = dim4;
  dims_[4].stride = stride4;

  int32 max_range = CheckAndFixDims();
  // The following should make sure that the full dimension
  // of the matrix has been "used up" by the tensor... this
  // is not an exhaustive check.
  if (mat.NumRows() <= 1) {
    KALDI_ASSERT(max_range == mat.NumCols());
  } else {
    KALDI_ASSERT(max_range == mat.NumRows() * mat.Stride());
  }
}

template<typename Real>
Tensor<Real>::Tensor(const VectorBase<Real> &vec,
               int32 dim1, int32 stride1,
               int32 dim2, int32 stride2,
               int32 dim3, int32 stride3,
               int32 dim4, int32 stride4,
               int32 dim5, int32 stride5):
    data_(const_cast<Real*>(vec.Data())) {
  dims_[0].dim = dim1;
  dims_[0].stride = stride1;
  dims_[1].dim = dim1;
  dims_[1].stride = stride1;
  dims_[2].dim = dim2;
  dims_[2].stride = stride2;
  dims_[3].dim = dim3;
  dims_[3].stride = stride3;
  dims_[4].dim = dim4;
  dims_[4].stride = stride4;

  int32 max_range = CheckAndFixDims();
  // The following should make sure that the full dimension
  // of the vector has been "used up" by the tensor... this
  // is not an exhaustive check.
  KALDI_ASSERT(max_range == vec.Dim());
}

template<typename Real>
int32 Tensor<Real>::CheckAndFixDims() {
  for (int32 i = 0; i < 5; i++) {
    KALDI_ASSERT(dims_[i].dim > 0);
    if (dims_[i].dim == 1) {
      dims_[i].stride = 0;
    } else {
      KALDI_ASSERT(dims_[i].stride != 0);
    }
  }

  std::vector<std::pair<int32, int32> > dims_tmp;
  for (int32 i = 0; i < 5; i++)
    if (dims_[i].dim != 1)
      dims_tmp.push_back(std::make_pair(dims_[i].stride, dims_[i].dim));
  
  // Sort from smallest to largest stride.
  std::sort(dims_tmp.begin(), dims_tmp.end());
  
  int32 largest_range = 0;
  
  for (size_t i = 1; i < dims_tmp.size(); i++) {
    int32 last_stride = dims_tmp[i-1].first,
        last_dim = dims_tmp[i-1].second,
        this_stride = dims_tmp[i].first,
        this_dim = dims_tmp[i].second;
    if (this_stride < last_dim * last_stride) {
      KALDI_ERR << "Invalid tensor dimensions (data would overlap): "
                << "dim" << (i-1) << " = " << last_dim
                << ", stride" << (i-1) << " = " << last_stride
                << ", dim" << i << " = " << this_dim;
    }
    largest_range = this_dim * this_stride;  // gets set to the last one.
  }
  return largest_range;
}


template<class Real>
void Tensor<Real>::Scale(BaseFloat alpha) {
  Real *data = data_;
  for (int32 i = 0; i < dims_[0].dim; i++, data += dims_[0].stride)
    for (int32 j = 0; j < dims_[1].dim; j++, data += dims_[1].stride)
      for (int32 k = 0; k < dims_[2].dim; k++, data += dims_[2].stride)
        for (int32 l = 0; l < dims_[3].dim; l++, data += dims_[3].stride)
          for (int32 m = 0; m < dims_[4].dim; m++, data += dims_[4].stride)
            *data *= alpha;
}  
  

struct AddDimInfo {
  int32 dim; // dim we count up to.
  int32 stride0;  // stride of *this.
  int32 stride1;  // stride of t1
  int32 stride2;  // stride of t2.
};

template<class Real>
void Tensor<Real>::AddTensorTensor(BaseFloat alpha,
                                   const Tensor<Real> &t1,
                                   const Tensor<Real> &t2,
                                   BaseFloat beta) {
  if (beta != 1.0) this->Scale(beta);

  AddDimInfo dim_info[5];
  
  for (int32 i = 0; i < 5; i++) {
    int32 d0 = dims_[i].dim, d1 = t1.dims_[i].dim, d2 = t2.dims_[i].dim;
    // Either dims should all be the same, or one should be one and the others
    // zero.
    if ( !(d0 == d1 && d1 == d2) &&
         !(d0 == 1 && d1 == d2) &&
         !(d1 == 1 && d0 == d2) &&
         !(d2 == 1 && d0 == d1)) {
      KALDI_ERR << "Dimension mismatch: for index " << i
                << ", dims are " << d0 << ", " << d1 << ", " << d2;
    }
    dim_info[i].dim = std::max(d0, std::max(d1, d2));
    dim_info[i].stride0 = dims_[i].stride;
    dim_info[i].stride1 = t1.dims_[i].stride;
    dim_info[i].stride2 = t2.dims_[i].stride;
  }
  // Remove trailing dimensions equal to one, then put them back on the front.
  // This means that any unnecessary loops are moved from the inside to the outside,
  // which is more efficient.
  for (int32 loop = 0; loop < 5; loop++) {
    if (dim_info[5 - 1].dim == 1) {
      AddDimInfo tmp = dim_info[5 - 1];
      for (int32 i = 1; i < 5; i++)
        dim_info[i] = dim_info[i - 1];
      dim_info[0] = tmp;
    }
  }

  Real *data0 = data_;
  const Real *data1 = t1.data_, *data2 = t2.data_;
  
  for (int32 j = 0; j < dim_info[0].dim; j++, data0 += dim_info[0].stride0,
           data1 += dim_info[0].stride1, data2 += dim_info[0].stride2)
    for (int32 k = 0; k < dim_info[1].dim; k++, data0 += dim_info[1].stride0,
             data1 += dim_info[1].stride1, data2 += dim_info[1].stride2)
      for (int32 l = 0; l < dim_info[2].dim; l++, data0 += dim_info[2].stride0,
               data1 += dim_info[2].stride1, data2 += dim_info[2].stride2)
        for (int32 m = 0; m < dim_info[3].dim; m++, data0 += dim_info[3].stride0,
                 data1 += dim_info[3].stride1, data2 += dim_info[3].stride2)
          for (int32 n = 0; n < dim_info[4].dim; n++, data0 += dim_info[4].stride0,
                   data1 += dim_info[4].stride1, data2 += dim_info[4].stride2)
            *data0 += alpha * *data1 * *data2;
}



// Explicit instantiation of the classes

template class Tensor<float>;
template class Tensor<double>;

} // namespace kaldi

