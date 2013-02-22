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

  void SetZero();  /// < Set to zero
  void SetUnit();  /// < Set to unit matrix.
  void SetRandn(); /// < Set to random values of a normal distribution
  void AddToDiag(Real r); ///< Add this quantity to the diagonal of the matrix.

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
  
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_rows_; }
  
  
  // Copy functions (do not resize).
  void CopyFromPacked(const CuPackedMatrix<Real> &src);
  void CopyFromPacked(const PackedMatrix<Real> &src);

  void Scale(Real c);
  
  void Read(std::istream &in, bool binary);
  
  void Write(std::ostream &out, bool binary) const;

  void Destroy();
  
  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(CuPackedMatrix<Real> *other);

  /// Swaps the contents of *this and *other.
  void Swap(PackedMatrix<Real> *other);
 protected:
  // Will only be called from this class or derived classes.
  void AddPacked(const Real alpha, const CuPackedMatrix<Real>& M);
  Real *data_;
  MatrixIndexT num_rows_;
 private:
  // Disallow assignment.
  PackedMatrix<Real> & operator =(const PackedMatrix<Real> &other);
}; // class CuPackedMatrix


/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuPackedMatrix<Real> &mat);


  
} // namespace


#endif
