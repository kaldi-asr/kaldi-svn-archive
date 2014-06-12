// cudamatrix/cu-tensor.h

// Copyright 2014     Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_TENSOR_H_
#define KALDI_CUDAMATRIX_CU_TENSOR_H_

#include "matrix/kaldi-tensor.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-value.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {


/**
   Note: the interface of class CuTensor is mostly the same as that of
   class Tensor, it's just the implementation that is different.
*/


template<class Real>
class CuTensor: public TensorBase<Real> {
 public:  
  CuTensor() { }
  
  /// A generic constructor
  CuTensor(const std::vector<std::pair<int32, int32> > &dims_strides,
           const CuMatrixBase<Real> &mat);
         
  /// A constructor that takes a raw data pointer.  Note: we accept
  /// "const" as a data pointer but this is not const-correct.
  CuTensor(const std::vector<std::pair<int32, int32> > &dims_strides,
           const Real *data);
  
  /// Convenience constructor for order-3 tensors.
  /// Note: in setting the strides, view them as strides on the raw data
  /// pointer, so if a particular index corresponds to the row-index of the
  /// matrix, its stride should be mat.Stride().
  /// Note: all dimensions must be >= 1.  Any stride for which the dimension
  /// is 1 will be set to zero.
  CuTensor(const CuMatrixBase<Real> &mat,
           int32 dim0, int32 stride0,
           int32 dim1, int32 stride1,
           int32 dim2, int32 stride2);
  
  /// Convenience constructor for order-4 tensors.
  CuTensor(const CuMatrixBase<Real> &mat,
           int32 dim0, int32 stride0,
           int32 dim1, int32 stride1,
           int32 dim2, int32 stride2,
           int32 dim3, int32 stride3);

  /// Convenience constructor for order-5 tensors.
  CuTensor(const CuMatrixBase<Real> &mat,
           int32 dim0, int32 stride0,
           int32 dim1, int32 stride1,
           int32 dim2, int32 stride2,
           int32 dim3, int32 stride3,
           int32 dim4, int32 stride4);


  /// Indexing operator
  inline Real& operator() (const std::vector<int32> &indexes);

  /// Indexing operator (const version)
  inline Real operator() (const std::vector<int32> &indexes) const;
  
  /// Does *this = t, element by element, possibly duplicating the contents of t across
  /// indices where *this has a dimension >1 and t has dimension 1.
  /// In the 3-dimensional case, for example, we do
  ///   (*this)(x,y,z) = t(x1,y1,z1)
  /// for all the indices x,y,z of this.  We compute the indices x1, y1 and z1 from
  /// x, y and z respectively, as follows.  Take one dimension (say, x, which
  /// is dimension zero).  If this->Dim(0) == t.Dim(0) then we let x1 = x.  If
  /// t.Dim(0) == 1, then we let x1 = 0, regardless of this->Dim(0).  Otherwise
  /// we crash.  
  /// Requires that *this is not aliased [c.f. CheckNoAliasing].
  void CopyFromTensor(const CuTensor<Real> &t);

  /// *this += alpha * t.
  ///
  /// Can either reduce (sum) or broadcast, or both.  For each index-position,
  /// the dimensions either must be the same, or one of them must be one.
  /// To be precise, it does this:
  /// for allowed pairs indexes1,indexes2:  *this(indexes1) += alpha * t(indexes2),
  /// where for each index-position p, 3 cases are allowed: either (i) the dimensions
  /// are equal (this->Dim(p) == t.Dim(p)),
  //// in which case we force that index to be the same, e.g. x1 == x2,
  /// or (ii) this->Dim(p) == 1, in which case we allow x1 = 0 and any x2,
  /// or (iii) t.Dim(p) == 1, in which case we allow any x1, and x2 = 0.
  void AddTensor(Real alpha,
                 const CuTensor<Real> &t);

  /// Does *this = alpha * t1 * t2 + beta * *this.
  ///
  ///  In general the code does as follows (we write it for a 3-dimensional
  /// tensor but the generalization is obvious).
  ///
  /// For some subset of indices (x,y,z,x1,y1,z1,x2,y2,z2), do:
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
  void AddTensorTensor(Real alpha,
                       const CuTensor<Real> &t1,
                       const CuTensor<Real> &t2,
                       Real beta);

  /// Does *this += alpha * Conv(t1 * t2), and related operations.
  ///
  /// This is defined as follows.  Like AddTensorTensor, in general the
  /// operation we're doing is:
  ///
  ///  (*this)(indices0) += alpha * t1(indices1) * t2(indices2),
  ///
  /// where the operation is done for some subset of the indices into
  /// *this, t1 and t2.  In cases that AddTensorTensor would accept,
  /// the behavior is exactly the same as AddTensorTensor, but this
  /// function handles one extra case.
  ///
  /// Let's consider the dimensions d0, d1, d2 for some arbitrary index
  /// position.  AddTensorTensor handles the case where d0, d1 and d2 are either
  /// all the same, or some of them are 1 and the rest are the same.  This
  /// function also handles the case where any two of them sum up to (the other
  /// one plus one), e.g. d0 + 1 == d1 + d2.  In that case, it limits
  /// the indexes that the operation is done for, to cases where
  /// x0 = x1 + x2, or, in general, (index that had largest dimension) to
  /// (sum of other indexes).
  void ConvTensorTensor(Real alpha,
                        const Tensor<Real> &t1,
                        const Tensor<Real> &t2);

  /// Scales each element of the tensor.  Note that aliasing does not stop this
  /// from working properly: for each distinct reachable memory location, it
  /// scales the data found there.
  void Scale(Real alpha);
  
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
  /// Taking the x (zeroth) dimension as an example, the subset of dimensions is
  /// as follows.
  ///   If this->Dim(0) == t1.Dim(0) == t2.Dim(0), then we limit the
  ///     sum to where x == x1 == x2.
  ///
  /// Otherwise, the behavior depends which of *this, t1 and t2 have
  /// the largest dimension.  We'll demonstrate it for where t2 is the
  /// largest, without intending any limitation.  We require that
  ///   this->Dim(0) + t1.Dim(0) == t2.Dim(0).  
  /// And we limit the sum to where x + x1 == x2.
  /// 
  /// Internally we turn this into an AddTensorTensor operation on a larger
  /// dimension of tensor (by adding index-positions).
  void ConvTensorTensor(Real alpha,
                        const CuTensor<Real> &t1,
                        const CuTensor<Real> &t2);

  
  /// Returns true if ((*this)-other).FrobeniusNorm()
  /// <= tol * (*this).FrobeniusNorm()
  bool ApproxEqual(const Tensor<Real> &other, float tol = 0.01) const;
  bool ApproxEqual(const CuTensor<Real> &other, float tol = 0.01) const;
  /// Use default copy and assignment operators.  

  // This struct is used by some functions that implement tensor operations, so
  // we make it public.
  struct DimInfo {
    int32 dim;
    int32 stride;
    DimInfo(int32 dim, int32 stride): dim(dim), stride(stride) { }
  };  

 protected:
  // The following two functions should only be called if we did not compile with CUDA
  // or could not get a CUDA card; in that case the contents are interpreted the
  // same as a regular tensor.
  inline const Tensor<Real> &GetTensor() const {
    return *(reinterpret_cast< const Tensor<Real>* >(this));
  }

  inline Tensor<Real> &GetTensor() {
    return *(reinterpret_cast<Tensor<Real>* >(this));
  }
 

 private:
  /// This is called from constructors.
  void Init(const std::vector<std::pair<int32, int32> > &dims_strides,
            const CuMatrixBase<Real> &mat);
  
  void Init(const std::vector<std::pair<int32, int32> > &dims_strides,
            const Real *data);

  
  /// This internal-only version of AddTensorTensor may reorder the
  /// tensor dimensions; it's called from the public AddTensorTensor
  /// after creating copies of the objects.
  void AddTensorTensor(Real alpha,
                       CuTensor<Real> *t1,
                       CuTensor<Real> *t2);
  
};



} // namespace

#endif
