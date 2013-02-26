#ifndef KALDI_CUDAMATRIX_CU_SP_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_SP_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/sp-matrix.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-packed-matrix.h"

namespace kaldi {

  typedef enum {
    kTakeLower,
    kTakeUpper,
    kTakeMean,
    kTakeMeanAndCheck
  } CuSpCopyType;

template<typename Real>
class CuSpMatrix : public CuPackedMatrix<Real> {
 public:
  // friendships
  //friend class std:vector<CuMatrix<Real> >;
  
  /// constructor
  CuSpMatrix(): CuPackedMatrix<Real>() {}
  
  explicit CuSpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
    : CuPackedMatrix<Real>(r, resize_type) {}

  explicit CuSpMatrix(const SpMatrix<Real> &orig)
    : PackedMatrix<Real>(orig) {}

  explicit CuSpMatrix(const CuSpMatrix<Real> &orig)
    : CuPackedMatrix<Real>(orig) {}

  /// deconstructor
  ~CuSpMatrix() {}

  /// resize
  inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    CuPackedMatrix<Real>::Resize(nRows, resize_type);
  }

  /// copyfromsp
  void CopyFromSp(const CuSpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }
  void CopyFromSp(const SpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }

  /// copy from Mat
#ifdef KALDI_PARANOID
  void CopyFromMat(const MatrixBase<Real> &orig,
		   CuSpCopyType copy_type = kTakeMeanAndCheck);
#else  // different default arg if non-paranoid mode.
  void CopyFromMat(const MatrixBase<Real> &orig,
		   CuSpCopyType copy_type = kTakeMean);
#endif

  // operators
  //inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    
  private:
  
};

} // namespace

#endif
