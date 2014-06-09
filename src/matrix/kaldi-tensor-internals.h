// matrix/kaldi-tensor-intenals.h

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

#ifndef KALDI_MATRIX_KALDI_TENSOR_INTERNALS_H_
#define KALDI_MATRIX_KALDI_TENSOR_INTERNALS_H_ 1

#include "matrix/matrix-common.h"
#include "matrix/kaldi-tensor.h"

namespace kaldi {
/// This header relates to the implementation of tensor operations for Matrix,
/// not CuMatrix.

/// Dimension info for 2 or 3-way tensor operations such as AddTensorTensor.
/// The "dim" here is the greater of all the dimensions (i.e.  if the dimensions
/// are, say 1, 1, 7 or 7, 7, 1, then it would be 7.  If any of the dimensions
/// are actually 1, we can tell because the corresponding stride would be zero.
struct TensorOperationDims {
  int32 dim;
  int32 stride_a;
  int32 stride_b;
  int32 stride_c;
  TensorOperationDims(int32 dim, int32 stride_a,
                      int32 stride_b, int32 stride_c):
      dim(dim), stride_a(stride_a), stride_b(stride_b), stride_c(stride_c) { }
  TensorOperationDims() { }
};

/// returns the sum of the elements in a vector 
template<typename Real> 
inline Real Xsum(const int N, const Real *X, const int incX) {
  Real sum = 0.0;
  for (int i =0; i < N; i++) { sum += X[i * incX];}
  return sum;
}
/// Removes any index-positions for which dim == 1 (even if
/// that would leave us with zero index-positions.
void ExciseUnitDims(std::vector<TensorOperationDims> *vec);


/// This function reorders the dimensions so that any "summed-over" dimensions
/// [i.e. for which stride_c == 0 and dim != 1] become last. 
/// We may later
/// put other heuristic reorderings inside this call too.
void ReorderSumToEnd(std::vector<TensorOperationDims> *vec);

/// @} end of \addtogroup matrix_group




/// This function constructs a tensor with c_data and the dim
/// and stride_c values of "dims", and calls Scale() on it.
/// It's a convenience function used in some of the internal code.
template<class Real>
void ScaleTensor(int32 num_indexes,
                 const TensorOperationDims *dims,
                 Real alpha,
                 Real *c_data);

/// This is called internally from Tensor::ScaleTensor().  Note: it does not
/// support tensors with aliasing or negative strides, and the calling code
/// takes care of this: the calling code removes any negative strides and
/// corrects the data pointer, and removes any aliasing that is easily removable
/// and checks that none remains.  Note that this function does not re-check
/// those preconditions.  This function will be most efficient if the smallest
/// strides are in the later index-positions.
template<class Real>
void ScaleTensor(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,
                 Real alpha,
                 Real *data);


/// This is a general-purpose, top-level form of the CopyTensor operation.
/// It uses the fields dim, stride_a and stride_b in "dims".
template<class Real>
void CopyTensorToplevel(const std::vector<TensorOperationDims> &dims,
                        const Real *a_data,
                        Real *b_data);



/// This is a top-level internal version of the function AddTensorTensor,
/// which does *c += alpha * *a.
template<class Real>
void AddTensorToplevel(const std::vector<TensorOperationDims> &dims,
                       Real alpha,
                       const Real *a_data,
                       Real *c_data);

/// This is a general-purpose, top-level form of the AddTensorTensor operation;
/// it first does reordering and then calls the appropriate function.
template<class Real>
void AddTensorTensorToplevel(const std::vector<TensorOperationDims> &dims,
                             Real alpha,
                             const Real *a_data,
                             const Real *b_data,
                             Real *c_data,
                             Real beta);

/**
   Returns true if, for c_data, the first dimension (dimension zero) of "dims"
   causes overlap. 
   We define this as follows [and note that we refer here only to c_data].  The
   first dimension causes overlap if for two different values of the first
   dimension's index (say, 0 vs 1), if we take the sets s_0 and s_1 of offsets
   in c_data that are accessible by ranging over all the values of the other
   indexes, there is overlap between s_0 and s_1.  In general, there would be
   overlap, and hence we would return true, if there is overlap between pair s_n
   and s_m, with m,n in the range (0, dim0).

   As a precondition, we require that there be no unit dims (dims[n].dim == 1).

   Some simple examples of this function returning true are:
      (i)   dims[0].stride_c = 0
      (ii)  dims[1].stride_c == dims[0].stride_c.
      (iii) dims[1].stride_c < dims[0].stride_c and
            dims[1].stride_c * dims[1].dim > dims[0].stride_c
   However, there are less obvious examples.
*/   
bool FirstDimOverlaps(int32 num_indexes,
                      const TensorOperationDims *dims);






}  // namespace kaldi


#endif  // KALDI_MATRIX_KALDI_TENSOR_INTERNALS_H_
