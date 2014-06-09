// matrix/kaldi-tensor.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2014  Pegah Ghahremani
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
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-tensor.h"
#include "matrix/kaldi-tensor-internals.h"

namespace kaldi {


template<typename Real>
int32 TensorBase<Real>::Dim(int32 index) const {
  KALDI_ASSERT(static_cast<size_t>(index) < dims_strides_.size());
  return dims_strides_[index].first;
}

template<typename Real>
int32 TensorBase<Real>::Stride(int32 index) const {
  KALDI_ASSERT(static_cast<size_t>(index) < dims_strides_.size());
  return dims_strides_[index].second;
}

template<typename Real>
void TensorBase<Real>::GetMinAndMaxOffset(int32 min_excluded_stride,
                                          int32 *min_offset,
                                          int32 *max_offset) const {
  *min_offset = 0;
  *max_offset = 0;
  for (int32 i = 0; i < dims_strides_.size(); i++) {
    int32 dim = dims_strides_[i].first, stride = dims_strides_[i].second;
    if (min_excluded_stride < 0 || abs(stride) < min_excluded_stride) {
      KALDI_ASSERT(dim >= 1);
      // Note: in current code, stride would always have to be >= 0.
      if (stride > 0) *max_offset += (dim - 1) * stride;
      else *min_offset += (dim - 1) * stride;
    }
  }
}

template<typename Real>
Tensor<Real>::Tensor(const MatrixBase<Real> &mat,
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
Tensor<Real>::Tensor(const MatrixBase<Real> &mat,
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
Tensor<Real>::Tensor(const MatrixBase<Real> &mat,
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
Tensor<Real>::Tensor(const std::vector<std::pair<int32, int32> > &dims_strides,
                     const MatrixBase<Real> &mat) {
  Init(dims_strides, mat);
}

template<typename Real>
Tensor<Real>::Tensor(const std::vector<std::pair<int32, int32> > &dims_strides,
                     const Real *data) {
  Init(dims_strides, data);
}

template<typename Real>
void Tensor<Real>::Init(const std::vector<std::pair<int32, int32> > &dims_strides,
                        const Real *data) {
  this->dims_strides_ = dims_strides;
  // Like SubMatrix, Tensor is not const-correct, as it would be too hard to
  // keep track of whether a tensor had been initialized with a const reference
  // or a pointer.
  this->data_ = const_cast<Real*>(data);
  this->CheckAndFixDims();
}

template<typename Real>
void Tensor<Real>::Init(const std::vector<std::pair<int32, int32> > &dims_strides,
                        const MatrixBase<Real> &mat) {
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


template<typename Real>
void TensorBase<Real>::CheckAndFixDims() {
  KALDI_ASSERT(!dims_strides_.empty());
  for (size_t i = 0; i < dims_strides_.size(); i++) {
    KALDI_ASSERT(dims_strides_[i].first >= 1);
    if (dims_strides_[i].first == 1)
      dims_strides_[i].second = 0;
  }
}

template<typename Real>
bool TensorBase<Real>::HasAliasing(const std::vector<int32> &ignore_dims) const {
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(dims_strides_.size());
  for (int32 i = 0; i < dims_strides_.size(); i++)
    if (std::find(ignore_dims.begin(), ignore_dims.end(), i) == ignore_dims.end())
      dims_strides.push_back(dims_strides_[i]);
  
  // use the absolute values of the strides.
  for (int32 i = 0; i < dims_strides.size(); i++)
    dims_strides[i].second = std::abs(dims_strides[i].second);
  
  // Sort from smallest to largest stride
  StrideLess sl;
  std::sort(dims_strides.begin(), dims_strides.end(), sl);
  
  for (size_t i = 0; i < dims_strides.size(); i++) {
    int32 dim = dims_strides[i].first,
        stride = dims_strides[i].second;
    if (dim != 1 && stride == 0)
      return true;  // Aliasing detected.
    if (i > 0 && dim != 1) {
      int32 prev_dim = dims_strides[i-1].first,
          prev_stride = dims_strides[i-1].second;
      if (stride < prev_stride * prev_dim)
        return true;  // aliasing detected
    }
  }
  return false;
}

template<typename Real>
void TensorBase<Real>::Flatten() {

  // Remove index-positions of "dims_strides" that have zero stride or unit dim,
  // since these wouldn't affect the set of reachable locations.  Make all
  // strides positive, and shift the data pointer to compensate for this change.
  int32 num_indexes = dims_strides_.size();
  std::vector<std::pair<int32, int32> > dims_strides_local;
  dims_strides_local.reserve(num_indexes);
  Real *new_data = data_;
  for (int32 i = 0; i < num_indexes; i++) {
    int32 dim = dims_strides_[i].first,
        stride = dims_strides_[i].second,
        abs_stride = std::abs(stride);
    if (dim > 1 && stride != 0)
      dims_strides_local.push_back(std::make_pair(dim, abs_stride));
    if (stride < 0) {
      // e.g. suppose stride = -2, and dim = 5.  The set of
      // reachable locations are 0, -2, ... -8.  We change stride
      // to abs_stride, and set new_data to data - 8, to keep
      // the set of reachable locations the same.
      new_data += stride * (dim - 1);
    }
  }

  // Sort from largest to smallest stride.
  StrideGreater sg;
  std::sort(dims_strides_local.begin(), dims_strides_local.end(), sg);

  for (int32 i = 1; i < dims_strides_local.size(); i++) {
    // merge pairs of index-positions that have the same stride.
    if (dims_strides_local[i-1].second == dims_strides_local[i].second) {
      dims_strides_local[i-1].first += dims_strides_local[i].first;
      dims_strides_local.erase(dims_strides_local.begin() + i);
      i--; // so we loop again for i.
    }
    // merge pairs of index-positions where the larger stride equals
    // the smaller (dim * stride).  Keep the product of dims, and the
    // smaller stride.
    if (dims_strides_local[i-1].second ==
        dims_strides_local[i].first * dims_strides_local[i].second) {
      dims_strides_local[i-1].first *= dims_strides_local[i].first;
      dims_strides_local[i-1].second = dims_strides_local[i].second;
      dims_strides_local.erase(dims_strides_local.begin() + i);
      i--; // so we loop again for i.
    }
  }
  if (dims_strides_local.empty())
    dims_strides_local.push_back(std::pair<int32, int32>(1, 0));
  dims_strides_ = dims_strides_local;
  data_ = new_data;
}


template<class Real>
void Tensor<Real>::AddTensorTensor(Real alpha,
                                   const Tensor<Real> &t1,
                                   const Tensor<Real> &t2,
                                   Real beta) {
  if (! (this->NumIndexes() == t1.NumIndexes() &&
         t1.NumIndexes() == t2.NumIndexes())) {
    int32 new_num_indexes = std::max(this->NumIndexes(),
                                     std::max(t1.NumIndexes(),
                                              t2.NumIndexes()));
    Tensor t0mod(new_num_indexes, *this),
        t1mod(new_num_indexes, t1),
        t2mod(new_num_indexes, t2);
    t0mod.AddTensorTensor(alpha, t1mod, t2mod, beta);
    return;
  }

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
                                       t1.dims_strides_[i].second,
                                  
                                  t2.dims_strides_[i].second,
                                       this->dims_strides_[i].second));
  }
  AddTensorTensorToplevel(dims, alpha, t1.Data(), t2.Data(), this->Data(),
                          beta);
}

template<class Real>
void Tensor<Real>::AddTensor(Real alpha,
                             const Tensor<Real> &t) {
  if (this->NumIndexes() != t.NumIndexes()) {
    int32 new_num_indexes = std::max(this->NumIndexes(),
                                     t.NumIndexes());
    Tensor this_mod(new_num_indexes, *this),
        t_mod(new_num_indexes, t);
    this_mod.AddTensor(alpha, t_mod);
    return;
  }
  
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
                                       t.dims_strides_[i].second,
                                       0,
                                       this->dims_strides_[i].second));
  }
  AddTensorToplevel(dims, alpha, t.Data(), this->Data());
}

template<typename Real>
void Tensor<Real>::Scale(Real alpha) {
  Tensor<Real> t(this->NumIndexes(), *this);  // Note: this uses the default copy constructor.
  t.Flatten();  // This removes aliasing.  It may fail if this is not possible
                // to do; this will only be the case for quite strange tensors,
                // and for now we just don't support scaling of such tensors.

  ScaleTensor(t.NumIndexes(),
              t.dims_strides_.empty() ? NULL : &(t.dims_strides_[0]),
              alpha, t.data_);
}

template<typename Real>
void Tensor<Real>::CopyFromTensor(const Tensor<Real> &t) {
  KALDI_ASSERT(this->NumIndexes() == t.NumIndexes());
  if (this->HasAliasing()) {
    KALDI_ERR << "Attempt to copy to a tensor that has aliasing.";
  }
  std::vector<TensorOperationDims> dims(this->NumIndexes());
  for (int32 i = 0; i < this->NumIndexes(); i++) {
    int32 dest_dim = this->dims_strides_[i].first,
        src_dim = t.dims_strides_[i].first,
        dest_stride= this->dims_strides_[i].second,
        src_stride = t.dims_strides_[i].second;
    if (!(src_dim == dest_dim || src_dim == 1)) {
      // We support broadcasting from dim 1 to a higher dim, or copying, but it
      // doesn't make sense to copy from many to one (it could sensibly imply
      // averaging or summing, but we're not going to get into that, as it
      // loses the meaning of copying).
      KALDI_ERR << "Invalid dimensions when copying tensor: for index-position "
                << i << ", source has dim " << src_dim << ", dest has dim "
                << dest_dim;
    }
    if (src_dim == 1)
      KALDI_ASSERT(src_stride == 0);  // We make sure of this in tensor
                                      // constructors.
    dims[i].dim = dest_dim;
    dims[i].stride_a = src_stride;
    dims[i].stride_b = dest_stride;
  }
  CopyTensorToplevel(dims, t.data_, this->data_);
}

template<class Real>
void Tensor<Real>::ConvTensorTensor(Real alpha,
                                    const Tensor<Real> &t1,
                                    const Tensor<Real> &t2) {
  if (! (this->NumIndexes() == t1.NumIndexes() &&
         t1.NumIndexes() == t2.NumIndexes())) {
    int32 new_num_indexes = std::max(this->NumIndexes(),
                                     std::max(t1.NumIndexes(),
                                              t2.NumIndexes()));
    Tensor t0mod(new_num_indexes, *this),
        t1mod(new_num_indexes, t1),
        t2mod(new_num_indexes, t2);
    t0mod.ConvTensorTensor(alpha, t1mod, t2mod);
    return;
  }
  
  // We'll create temporary tensors that we can just call AddTensorTensor on,
  // by adding a new index-position for each dimension that is convolved.
  std::vector<std::pair<int32, int32 > > dims_strides0, dims_strides1,
      dims_strides2;
  int32 num_indexes = this->NumIndexes();
  dims_strides0.reserve(num_indexes * 2);
  dims_strides1.reserve(num_indexes * 2);
  dims_strides2.reserve(num_indexes * 2);
  for (int32 i = 0; i < num_indexes; i++) {
    int32 d0 = this->Dim(i), d1 = t1.Dim(i), d2 = t2.Dim(i),
         s0 = this->Stride(i), s1 = t1.Stride(i), s2 = t2.Stride(i);
    int32 d_max = std::max(d0, std::max(d1, d2));
    if ((d0 == 1 || d0 == d_max) && (d1 == 1 || d1 == d_max) &&
        (d2 == 1 || d2 == d_max)) {
      // This is the normal case that could be handled by AddTensorTensor;
      // leave these dimensions unchanged.
      dims_strides0.push_back(std::pair<int32, int32>(d0, s0));
      dims_strides1.push_back(std::pair<int32, int32>(d1, s1));
      dims_strides2.push_back(std::pair<int32, int32>(d2, s2));
    } else {
      // We can handle the case where one dimension + 1 == (sum of other 2).
      // In each case we add two indexes, one corresponding to the dimension of
      // each of the smaller dims.
      if (d0 + 1 == d1 + d2) {
        dims_strides0.push_back(std::pair<int32, int32>(d1, s0));
        dims_strides0.push_back(std::pair<int32, int32>(d2, s0));
        dims_strides1.push_back(std::pair<int32, int32>(d1, s1));
        dims_strides1.push_back(std::pair<int32, int32>(1, 0));
        dims_strides2.push_back(std::pair<int32, int32>(1, 0));
        dims_strides2.push_back(std::pair<int32, int32>(d2, s2));
      } else if (d1 + 1 == d0 + d2) {
        dims_strides0.push_back(std::pair<int32, int32>(d0, s0));
        dims_strides0.push_back(std::pair<int32, int32>(1, 0));
        dims_strides1.push_back(std::pair<int32, int32>(d0, s1));
        dims_strides1.push_back(std::pair<int32, int32>(d2, s1));
        dims_strides2.push_back(std::pair<int32, int32>(1, 0));
        dims_strides2.push_back(std::pair<int32, int32>(d2, s2));
      } else if (d2 + 1 == d0 + d1) {
        dims_strides0.push_back(std::pair<int32, int32>(d0, s0));
        dims_strides0.push_back(std::pair<int32, int32>(1, 0));
        dims_strides1.push_back(std::pair<int32, int32>(1, 0));
        dims_strides1.push_back(std::pair<int32, int32>(d1, s1));
        dims_strides2.push_back(std::pair<int32, int32>(d0, s2));
        dims_strides2.push_back(std::pair<int32, int32>(d1, s2));
      } else {
        KALDI_ERR << "This combination of dimensions cannot be handled: "
                  << "for index-position " << i
                  << ", (d0, d1, d2) = (" << d0 << ", " << d1 << ", " << ")";
      }
    }
  }
  Tensor<Real> t0mod(dims_strides0, this->Data()),
      t1mod(dims_strides1, t1.Data()),
      t2mod(dims_strides2, t2.Data());
  t0mod.AddTensorTensor(alpha, t1mod, t2mod, 1.0);
}
template<class Real>
Tensor<Real>::Tensor(int32 new_order,
                     const Tensor<Real> &tensor) {
  KALDI_ASSERT(new_order >= static_cast<int32>(tensor.NumIndexes()));
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(new_order);
  for (int32 i = 0; i < new_order - tensor.NumIndexes(); i++)
    dims_strides.push_back(std::pair<int32, int32>(1, 0));
  dims_strides.insert(dims_strides.end(),
                      tensor.dims_strides_.begin(),
                      tensor.dims_strides_.end());
  
  this->Init(dims_strides, tensor.data_);
}

template<class Real>
Tensor<Real>::Tensor(const Tensor<Real> &tensor) {
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(tensor.NumIndexes());
  dims_strides.insert(dims_strides.end(), tensor.dims_strides_.begin(),
                      tensor.dims_strides_.end()); 
  int32 min_offset, max_offset;
  tensor.GetMinAndMaxOffset(-1, &min_offset, &max_offset);
  void *data;
  void *temp;
  size_t size = static_cast<size_t>(max_offset+1) * sizeof(Real);
  if (NULL != (data = KALDI_MEMALIGN(16, size, &temp))) 
    this->data_ = static_cast<Real *> (data);
  this->dims_strides_ = dims_strides;
  this->CopyFromTensor(tensor);
}

template<typename Real> 
Real Tensor<Real>::FrobeniusNorm() const {
  typedef std::pair<int32, int32> DimsStrides;
  Matrix<Real> tmp(1,1);
  std::vector<DimsStrides> norm_dims_strides;
  norm_dims_strides.push_back(DimsStrides(1,2));
  Tensor<Real> t_norm(norm_dims_strides, tmp);
  t_norm.AddTensorTensor(static_cast<Real>(1.0), *this, *this, static_cast<Real>(0.0));
  return std::sqrt(*t_norm.Data());
}

template<typename Real> 
Real Tensor<Real>::Sum() const {
  typedef std::pair<int32, int32> DimsStrides; 
  std::vector<DimsStrides> dims_strides;
  dims_strides.push_back(DimsStrides(1,1));
  Matrix<Real> mat(1, 1);
  Tensor<Real> temp(dims_strides, mat);
  temp.AddTensor(1.0, *this);
  return (*temp.Data());
}

template<typename Real> 
bool Tensor<Real>::ApproxEqual(const Tensor<Real> &other, float tol) const {
  // Check dimensions and strides mismatch
  if (this->NumIndexes() != other.NumIndexes()) 
    KALDI_ERR << "ApproxEqual: size mismatch.";
  for (int i = 0; i < this->NumIndexes(); i++)  { 
    if ( this->Stride(i) != other.Stride(i) || this->Dim(i) != other.Dim(i))
      KALDI_ERR << "ApproxEqual: size mismatch.";
  }
  Tensor<Real> tmp(*this);
  tmp.AddTensor(-1.0, other);
  return (tmp.FrobeniusNorm() <= static_cast<Real>(tol) *
          this->FrobeniusNorm());
}


template class TensorBase<float>;
template class TensorBase<double>;


template class Tensor<float>;
template class Tensor<double>;


} // namespace kaldi

