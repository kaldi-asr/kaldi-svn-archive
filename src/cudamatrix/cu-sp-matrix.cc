#if HAVE_CUDA==1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-math.h"
#include "cu-sp-matrix.h"

namespace kaldi {

/**
 * C++ templated wrapper of ANSI-C CUBLAS function SPR
 */
#if HAVE_CUDA==1
template<typename Real> inline void cublas_spr(char uplo, int n, const Real alpha, const Real* x, int incx, Real* AP) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_spr<float>(char uplo, int n, const float alpha, const float*x, int incx, float* AP) {
  cublasSspr(uplo, n, alpha, x, incx, AP);
}
template<> inline void cublas_spr<double>(char uplo, int n, const double alpha, const double* x, int incx, double* AP) {
  cublasDspr(uplo, n, alpha, x, incx, AP);
}
#endif


template<>
void CuSpMatrix<float>::AddVec2(const float alpha, const CuVectorBase<float> &v, char uplo) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
  cublas_spr((uplo=='U'?'U':'L'), v.Dim(), alpha, v.Data(), 1, this->data_);
}

template<>
void CuSpMatrix<double>::AddVec2(const double alpha, const CuVectorBase<double> &v, char uplo) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
  cublas_spr((uplo=='U'?'U':'L'), v.Dim(), alpha, v.Data(), 1, this->data_);
}

  /*
  template<class Real>
  Real CuSpMatrix<Real>::Trace() const {
    Real* ans;
#if HAVE_CUDA==1
    if (CuDevice::Instantiate().Enabled()) {
      Timer tim;

      int dimBlock(CUBLOCK);
      int dimGrid(n_blocks((*this).NumRows(), CUBLOCK));

      cuda_trace(dimGrid, dimBlock, this->data_, ans);
      CU_SAFE_CALL(cudaGetLastError());

      CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    } else
#endif
      {
	ans = (*this).Mat().Trace();
      }
    return ans;
  }
  */
template class CuSpMatrix<float>;
template class CuSpMatrix<double>;

} // namespace
