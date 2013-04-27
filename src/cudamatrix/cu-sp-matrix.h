#ifndef KALDI_CUDAMATRIX_CU_SP_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_SP_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/sp-matrix.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {

template<typename Real>
class CuSpMatrix : public CuPackedMatrix<Real> {
 public:
  
  CuSpMatrix(): CuPackedMatrix<Real>() {}
  
  explicit CuSpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
    : CuPackedMatrix<Real>(r, resize_type) {}

  explicit CuSpMatrix(const SpMatrix<Real> &orig)
    : CuPackedMatrix<Real>(orig) {}

  explicit CuSpMatrix(const CuSpMatrix<Real> &orig)
    : CuPackedMatrix<Real>(orig) {}

  explicit CuSpMatrix(const CuMatrixBase<Real> &orig,
                      SpCopyType copy_type = kTakeLower)
      : CuPackedMatrix<Real>(orig.NumRows(), kUndefined) {
    CopyFromMat(orig, copy_type);
  }

  ~CuSpMatrix() {}  

  inline const SpMatrix<Real> &Mat() const {
    return *(reinterpret_cast<const SpMatrix<Real>* >(this));
  }

  inline SpMatrix<Real> &Mat() {
    return *(reinterpret_cast<SpMatrix<Real>* >(this));
  }
  
  inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    CuPackedMatrix<Real>::Resize(nRows, resize_type);
  }

  void CopyFromSp(const CuSpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }
  void CopyFromSp(const SpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }

  void CopyFromMat(const CuMatrixBase<Real> &orig,
                   SpCopyType copy_type = kTakeLower);
  
  void CopyToSp(SpMatrix<Real> *dst) {
    CuPackedMatrix<Real>::CopyToMat(dst);
  }
  
  void Invert(Real *logdet = NULL, Real *det_sign = NULL,
              bool inverse_needed = true);

  void AddVec2(const Real alpha, const CuVectorBase<Real> &v);

  void AddMat2(const Real alpha, const CuMatrixBase<Real> &M,
               MatrixTransposeType transM, const Real beta);
  
};

} // namespace

#endif
