// cudamatrix/cu-packed-matrix.h

// Copyright 2009-2013  Johns Hopkins University (author: Daniel Povey)
//                      Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CU_PACKED_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_PACKED_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/packed-matrix.h"
#include "matrix/sp-matrix.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {


/**
 * Matrix for CUDA computing.  This is a base class for packed
 * triangular and symmetric matrices. 
 * Does the computation on the CUDA card when CUDA is compiled in and
 * we have a suitable GPU (CuDevice::Instantiate().Enabled() == true);
 * otherwise, does it on the CPU.
 */


/// @brief Packed CUDA matrix: base class for triangular and symmetric matrices on
///        a GPU card.
template<typename Real>
class CuPackedMatrix {
 public:
  friend class CuMatrixBase<Real>;
  friend class CuVectorBase<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuRand<Real>;

  
  CuPackedMatrix() : data_(NULL), num_rows_(0) {}

  explicit CuPackedMatrix(MatrixIndexT r,
                          MatrixResizeType resize_type = kSetZero):
      data_(NULL), num_rows_(0) {  Resize(r, resize_type);  }
  
  explicit CuPackedMatrix(const PackedMatrix<Real> &orig) : data_(NULL) {
    Resize(orig.NumRows(), kUndefined);
    CopyFromPacked(orig);
  }

  explicit CuPackedMatrix(const CuPackedMatrix<Real> &orig) : data_(NULL) {
    Resize(orig.NumRows(), kUndefined);
    CopyFromPacked(orig);
  }

  // The following two functions should only be called if we did not compile with CUDA
  // or could not get a CUDA card; in that case the contents are interpreted the   
  // same as a regular matrix.                                                     
                 
  inline const MatrixBase<Real> &Mat() const {
    return *(reinterpret_cast<const MatrixBase<Real>* >(this));
  }
  inline MatrixBase<Real> &Mat() {
    return *(reinterpret_cast<MatrixBase<Real>* >(this));
  }
  
  /// Dimensions
  ::MatrixDim Dim() const {
    ::MatrixDim d = {num_rows_, num_cols_, stride_};
    return d;
  }
  
  MatrixIndexT Stride() const { return stride_; }

  void SetZero();  /// < Set to zero
  void Set(Real value);
  void SetUnit();  /// < Set to unit matrix.
  void SetRandn(); /// < Set to random values of a normal distribution
  void AddToDiag(Real r); ///< Add this quantity to the diagonal of the matrix.
  void Add(Real value);

  void ApplyLog();
  /// Multiply two matrices elementhwise: C = A .* C 
  void MulElements(const CuPackedMatrix<Real>& A);
  /// scale i'th column by scale[i] 
  void MulColsVec(const CuVectorBase<Real>& scale);
  /// scale i'th row by scale[i] 
  void MulRowsVec(const CuVectorBase<Real>& scale);
  /// divide i'th row by scale[i]
  void DivRowsVec(const CuVectorBase<Real>& div);
  /// B = aplha * A + beta * B
  void AddMat(Real alpha, const CuPackedMatrix<Real>& A, Real beta=1.0);
  /// B = aplha * row + beta * B 
  void AddVecToCols(Real alpha, const CuVectorBase<Real>& col, Real beta=1.0);
  /// B = aplha * row + beta * B
  void AddVecToRows(Real alpha, const CuVectorBase<Real>& row, Real beta=1.0);
  Real Trace() const;

  ~CuPackedMatrix() { Destroy(); }

  /// Set packed matrix to a specified size (can be zero).
  /// The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero);
  
  // Copy functions (do not resize).
  void CopyFromPacked(const CuPackedMatrix<Real> &src);
  void CopyFromPacked(const PackedMatrix<Real> &src);
  void CopyFromMat(const Matrix<Real> &src);

  void CopyToMat(PackedMatrix<Real> *dst) const;
  void CopyToMat(Matrix<Real> *dst) const;

  void Scale(Real c);
  
  void Read(std::istream &in, bool binary);
  
  void Write(std::ostream &out, bool binary) const;

  void Destroy();
  
  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(CuPackedMatrix<Real> *other);

  /// Swaps the contents of *this and *other.
  void Swap(PackedMatrix<Real> *other);

  void Swap(Matrix<Real> *other);

  Real* Data() { return data_; }
  const Real* Data() const { return data_; }
  /// Size
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_rows_; }
  /// Returns size in bytes of the data held by the matrix.
  size_t  SizeInBytes() const {
    return static_cast<size_t>(num_rows_) * static_cast<size_t>(stride_) * sizeof(Real);
  }

 protected:
  // Will only be called from this class or derived classes.
  void AddPacked(const Real alpha, const CuPackedMatrix<Real>& M);
  Real *data_;
  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;
  MatrixIndexT stride_;
 private:
  // Disallow assignment.
  PackedMatrix<Real> & operator =(const PackedMatrix<Real> &other);
}; // class CuPackedMatrix


/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuPackedMatrix<Real> &mat);


  
} // namespace


#endif
