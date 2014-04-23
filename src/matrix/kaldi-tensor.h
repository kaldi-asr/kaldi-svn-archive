// matrix/kaldi-tensor.h

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

#ifndef KALDI_MATRIX_KALDI_TENSOR_H_
#define KALDI_MATRIX_KALDI_TENSOR_H_ 1

#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {


/// \addtogroup matrix_group
/// @{



/// This class can represent a up to a 5 dimensional tensor; we can increase
/// this later if needed.  Tensors of lower dimension are represented by setting
/// the higher-numbered dimensions to one.
template<class Real>
class Tensor {  
  Real *Data() { return data_; }
  
  /// Rather than having rows and columns, the tensor has numbered dimensions.
  /// 0 <= index < 5.
  int32 Dim(int32 index) const;

  /// 0 <= index < 5.
  int32 Stride(int32 index) const;


  /// Note: in setting the strides, view them as strides on the raw data
  /// pointer, so if a particular index corresponds to the row-index of the
  /// matrix, its stride should be mat.Stride().
  Tensor(const MatrixBase<Real> &mat,
         int32 dim0, int32 stride0,
         int32 dim1, int32 stride1,
         int32 dim2, int32 stride2,
         int32 dim3 = 1, int32 stride3 = 0,
         int32 dim4 = 1, int32 stride4 = 0);

  Tensor(const VectorBase<Real> &vec,
         int32 dim0, int32 stride0,
         int32 dim1, int32 stride1,
         int32 dim2, int32 stride2,
         int32 dim3 = 1, int32 stride3 = 0,
         int32 dim4 = 1, int32 stride4 = 0);
  
  inline Real &operator () (int32 d0, int32 d1, int32 d2, int32 d3 = 0, int32 d4 = 0);

  inline Real operator () (int32 d0, int32 d1, int32 d2, int32 d3 = 0, int32 d4 = 0) const;
  
  /// Does *this = alpha * t1 * t2 + beta * *this.
  ///
  ///  In general the code does as follows (we write it for a 3-dimensional
  /// tensor but the generalization is obvious).
  /// (*this) *= beta.
  /// Then, for some subset of indices (x,y,z,x1,y1,z1,x2,y2,z2), do:
  ///   (*this)(x,y,z) += alpha * t1(x1,y1,z1) * t2(x2,y2,z2)
  /// 
  /// The dimensions determine which subset of indices we take.  x,y and z all
  /// behave the same, so we take x as an example, noting that x is dimension zero.
  /// If this->Dim(0) == t1.Dim(0) == t2.Dim(0), then limit the
  /// subset to where x == x1 == x2.
  /// If, from *this, t1, and t2, any two of Dim(0) are the same and the
  /// other is one, then limit the subset to where those two are the same.
  /// If neither of the two cases above apply, then crash.
  /// 
  /// This is just a special case of ConvTensorTensor, and we currently
  /// implement it by calling that function.
  void AddTensorTensor(BaseFloat alpha,
                       const Tensor<Real> &t1,
                       const Tensor<Real> &t2,
                       BaseFloat beta);
  
  /// Does *this = alpha * Conv(t1 * t2) + beta * *this.
  /// Conv(t1 * t2) is a generic kind of convolution, of the type required
  /// by convolutional neural networks (see papers by Yann LeCun)... I'd say it's
  /// more similar to correlation than convolution as there is no sign flip, but
  /// we're calling it convolution.  To understand it, first read the comment for
  /// AddTensorTensor (above), because this is only a small
  /// extension of that.
  ///
  /// In general the code does as follows:
  /// (*this) *= beta.
  /// Then, for some subset of indices (x,y,z,x1,y1,z1,x2,y2,z2), do:
  ///   (*this)(x,y,z) += alpha * t1(x1,y1,z1) * t2(x2,y2,z2)
  ///
  /// Taking the x dimension as an example, the subset of dimensions is
  /// as follows.   Noting that x is dimension zero,
  ///   If this->Dim(0) == t1.Dim(0) == t2.Dim(0), then we limit the
  ///     sum to where x == x1 == x2.
  ///
  /// Otherwise, the behavior depends which of *this, t1 and t2 have
  /// the largest dimension.  We'll demonstrate it for where t2 is the
  /// largest, without intending any limitation.  We require that
  ///   this->Dim(0) + t1.Dim(0) == t2.Dim(0).  
  /// And we limit the sum to where x + x1 == x2.
  void ConvTensorTensor(BaseFloat alpha,
                        const Tensor<Real> &t1,
                        const Tensor<Real> &t2,
                        BaseFloat beta);

  /// Does *this = *this * alpha.  Ideally you should never call this, even
  /// implicitly, as it's not very optimized; better to scale the underlying
  /// matrix or vector.
  void Scale(BaseFloat alpha);

  /// returns a number from 1 to 5, which equals 1 + the largest
  /// value i < 5 such that Dim(i) > 1.  Basically, the number of
  /// non-trivial dimensions the tensor has.
  int32 NumDims();
  
  // Use default copy and assignment operators.  
 private:
  int32 CheckAndFixDims();  // Check dimensions, and sets the stride of any dims with
                            // dimension==1, to zero.  Returns largest (stride*dim),
                            // which can be used as part of a check in the
                            // initializer.
  
  
  Tensor() { } // Disallow default constructor.
  
  Real *data_;  // not owned here.
  struct DimInfo {
    int32 dim;
    int32 stride;
  };
  DimInfo dims_[5];
};


/// @} end of \addtogroup matrix_group


}  // namespace kaldi

#include "matrix/kaldi-tensor-inl.h"


#endif  // KALDI_MATRIX_KALDI_TENSOR_H_
