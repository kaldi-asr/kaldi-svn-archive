// matrix/kaldi-tensor.h

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

#ifndef KALDI_MATRIX_KALDI_TENSOR_H_
#define KALDI_MATRIX_KALDI_TENSOR_H_ 1

#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {


/// \addtogroup matrix_group
/// @{


/** Class TensorBase is used as a base class to make sure certain things stay in
    sync between classes Tensor and CuTensor, and to share some code.  It is not
    intended to be used directly by the user.
 */
template<class Real>
class TensorBase {
 public:
  Real *Data() { return data_; }

  const Real *Data() const { return data_; }
  
  /// Returns the number of indices.
  inline int32 NumIndexes() const { return dims_strides_.size(); }
  
  /// Rather than having rows and columns, the tensor has numbered dimensions.
  /// 0 <= index < NumIndexes()
  int32 Dim(int32 index) const;

  /// 0 <= index < NumIndexes()
  int32 Stride(int32 index) const;

  TensorBase(): data_(NULL) { } 

  /// Returns true if there is 'aliasing' in the tensor.  By 'aliasing' we mean
  /// that the same memory location can be accessed via multiple distinct
  /// indices.  This check is only necessary when we are going to write to a
  /// tensor.  The simplest causes of aliasing are having two separate
  /// dimensions with the same stride and dimensions not equal to one, or
  /// having dimensions with zero stride and dimension not equal to one, but
  /// there are other ways aliasing can be present.  
  ///
  /// If allow_trailing_zero_stride == false, we allow zero strides
  /// (where dim != 1) for trailing index-positions, e.g. in a 3-d tensor
  /// the last element could have zero stride and non-unit dim.  This is
  /// encountered when we need to sum over some dimensions.
  bool HasAliasing() const { return HasAliasing(std::vector<int32>()); }


  /// This modifies the structure of the tensor to remove as many index
  /// positions as possible while keeping the set of reachable locations
  /// the same.  This is mostly not to be called by the user, but by
  /// things internal to the Tensor class. 
  void Flatten();
  
 protected:

  /// Outputs the data offset that we'd get if all dims were set at their
  /// minimum/maximum value, but only considers dimensions whose absolute stride
  /// is less than "min_excluded_stride", if min_excluded_stride >= 0.  This is
  /// useful in checks.  min_offset will normally be zero, but could be negative
  /// if strides are negative (which is not normal).
  void GetMinAndMaxOffset(int32 min_excluded_stride,
                          int32 *min_offset,
                          int32 *max_offset) const;


  /// Check dimensions are >= 1, and sets the stride of any index-position with
  /// dimension==1, to zero.  These strides are "don't-care" values for most
  /// purposes, but in HasAliasing(), we rely on the fact that the strides for
  /// those index positions are zero [it affects the sorting of the
  /// index-positions].
  void CheckAndFixDims(); 
  
  /// This version of HasAliasing allows you to select some dimensions to ignore
  /// for purposes of checking whether there is aliasing.
  bool HasAliasing(const std::vector<int32> &ignore_dims) const;
  
  // Sorting operators for (dim, stride) pairs
  struct StrideGreater {
    inline bool operator () (const std::pair<int32, int32> &d1,
                             const std::pair<int32, int32> &d2) {
      return (std::abs(d1.second) > std::abs(d2.second));
    }
  };
  struct StrideLess {
    inline bool operator () (const std::pair<int32, int32> &d1,
                             const std::pair<int32, int32> &d2) {
      return (std::abs(d1.second) < std::abs(d2.second));
    }
  };
  
  // These data members must stay in sync with those
  Real *data_;  // the pointer is not owned here.
  std::vector<std::pair<int32, int32> > dims_strides_;
  
};



/**
    Tensor is a generalized multi-dimensional array, in the normal case.
    However, it is a little more than that because it is possible for the same
    data to be accessed by different indexes.  Suppose we have a 4-d tensor: in
    general, each of the 4 index-positions, has a dimension and a stride. [A
    stride is the amount by which the pointer moves when the index value changes
    by one.]  It is possible for some of the strides to be the same, which makes
    it possible to construct a tensor that corresponds to blocks that can slide
    around a matrix.  Even negative strides are allowed in general (although for
    this, you'd have to use the constructor from Real*, not from
    MatrixBase<Real>).
    
    This class does not 'own' its data; it provides a 'look' into data owned
    by a Matrix (we may also extend it to 'look' into data owned by a Vector).
    You can initialize it with
\code
    Tensor tensor(matrix, dims...);
\endcode    
    where dims is a vector of pairs (dimension, stride), or
\code    
    Tensor tensor;
    ...
    tensor = Tensor(matrix, dims...);
\endcode

   All tensor operations, such as AddTensorTensor, require that the number of
   index-positions be the same.  If the number of index-positions differs, we'll
   resolve it by prepending (dim=1, stride=0) to the shorter tensor or tensors
   until they match, and then re-try the operation.
*/

template<class Real>
class Tensor: public TensorBase<Real> {
 public:  
  Tensor() { }
  
  /// A generic constructor
  Tensor(const std::vector<std::pair<int32, int32> > &dims_strides,
         const MatrixBase<Real> &mat);
         
  /// A constructor that takes a raw data pointer.  Note: we accept
  /// "const" as a data pointer but this is not const-correct.
  Tensor(const std::vector<std::pair<int32, int32> > &dims_strides,
         const Real *data);

  /// This constructor copies the tensor and possibly increases the number of
  /// indices by prepending extra index-positions with (dim = 1, stride = 0).
  /// It's used to force tensor orders to match in operations like
  /// AddTensorTensor.  The new tensor will reference the same underlying data.
  Tensor(int32 new_num_indices,
         const Tensor<Real> &tensor);

  /// this is  a copy constructor from Tensor class
  /// It allocates new memory
  Tensor(const Tensor<Real> &tensor);

  /// This constructor can be used to modify the order of the tensor indices.
  /// Index i of *this will correspond to index "index_order[i]" of "tensor".
  Tensor(const std::vector<int32> index_order,
         const Tensor<Real> &tensor);
  /// Convenience constructor for order-3 tensors.
  /// Note: in setting the strides, view them as strides on the raw data
  /// pointer, so if a particular index corresponds to the row-index of the
  /// matrix, its stride should be mat.Stride().
  /// Note: all dimensions must be >= 1.  Any stride for which the dimension
  /// is 1 will be set to zero.
  Tensor(const MatrixBase<Real> &mat,
         int32 dim0, int32 stride0,
         int32 dim1, int32 stride1,
         int32 dim2, int32 stride2);
  
  /// Convenience constructor for order-4 tensors.
  Tensor(const MatrixBase<Real> &mat,
         int32 dim0, int32 stride0,
         int32 dim1, int32 stride1,
         int32 dim2, int32 stride2,
         int32 dim3, int32 stride3);

  /// Convenience constructor for order-5 tensors.
  Tensor(const MatrixBase<Real> &mat,
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
  /// for all the indices x,y,z of *this.  We compute the indices x1, y1 and z1 from
  /// x, y and z respectively, as follows.  Take one dimension (say, x, which
  /// is dimension zero).  If this->Dim(0) == t.Dim(0) then we let x1 = x.  If
  /// t.Dim(0) == 1, then we let x1 = 0, regardless of this->Dim(0).  Otherwise
  /// we crash.  
  /// Requires that *this is not aliased [c.f. CheckNoAliasing].
  void CopyFromTensor(const Tensor<Real> &t);

  /// *this += alpha * t.
  ///
  /// Can either reduce (sum) or broadcast, or both. For each index-position,
  /// the dimensions either must be the same, or one of them must be one.
  /// To be precise, it does this:
  /// for allowed pairs indexes1,indexes2:  *this(indexes1) += alpha * t(indexes2),
  /// where for each index-position p, 3 cases are allowed: either (i) the dimensions
  /// are equal (this->Dim(p) == t.Dim(p)),
  //// in which case we force that index to be the same, e.g. x1 == x2,
  /// or (ii) this->Dim(p) == 1, in which case we allow x1 = 0 and any x2,
  /// or (iii) t.Dim(p) == 1, in which case we allow any x1, and x2 = 0.
  void AddTensor(Real alpha,
                 const Tensor<Real> &t);

  /// Does *this = alpha * t1 * t2 + beta * *this.
  ///
  ///  In general the code does as follows (we write it for a 3-dimensional
  /// tensor but the generalization is obvious).
  ///
  /// For some subset of indices (x,y,z,x1,y1,z1,x2,y2,z2), do:
  ///   (*this)(x,y,z) += alpha * t1(x1,y1,z1) * t2(x2,y2,z2)
  /// 
  /// The dimensions determine which subset of indices we take.
  /// Suppose we took 3 tensors [*this, t1 and t2] that all had
  /// the same dimensions, and then replaced some of the dimensions
  /// with a 1.  AddTensorTensor will accept all such combinations,
  /// and the indexes of all the corresponding dimensions must match
  /// (e.g. x == x1 == x2), except those whose dimensions were replaced
  /// with 1, which will be zero.
  ///
  void AddTensorTensor(Real alpha,
                       const Tensor<Real> &t1,
                       const Tensor<Real> &t2,
                       Real beta);

  /// Scales each element of the tensor.  Note that aliasing does not stop this
  /// from working properly: for each distinct reachable memory location, it
  /// scales the data found there.
  void Scale(Real alpha);
  
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
  /// Frobenius norm, which is the sqrt of sum of square elements of tensor
  Real FrobeniusNorm() const;
  
  /// Returns sum of all elements of Tensor
  Real Sum() const;
  /// Returns true if ((*this)-other).FrobeniusNorm()
  /// <= tol * (*this).FrobeniusNorm()
  bool ApproxEqual(const Tensor<Real> &other, float tol = 0.01) const;

  /// Use default copy and assignment operators.  

  // This struct is used by some functions that implement tensor operations, so
  // we make it public.
  struct DimInfo {
    int32 dim;
    int32 stride;
    DimInfo(int32 dim, int32 stride): dim(dim), stride(stride) { }
  };  
 private:
  /// This is called from constructors.
  void Init(const std::vector<std::pair<int32, int32> > &dims_strides,
            const MatrixBase<Real> &mat);
  
  void Init(const std::vector<std::pair<int32, int32> > &dims_strides,
            const Real *data);

  
  /// This internal-only version of AddTensorTensor may reorder the
  /// tensor dimensions; it's called from the public AddTensorTensor
  /// after creating copies of the objects.
  void AddTensorTensor(Real alpha,
                       Tensor<Real> *t1,
                       Tensor<Real> *t2);
  
};


/// @} end of \addtogroup matrix_group


}  // namespace kaldi

#include "matrix/kaldi-tensor-inl.h"


#endif  // KALDI_MATRIX_KALDI_TENSOR_H_
