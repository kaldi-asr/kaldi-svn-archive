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
#include "cu-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"

namespace kaldi {

template<typename Real>
void CuTpMatrix<Real>::Cholesky(const CuSpMatrix<Real> &orig) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    CuMatrix<Real> tmp(orig);
    tmp.Cholesky();
    this->CopyFromMat(tmp, kNoTrans);
  } else
#endif
  {

  }
}

#if HAVE_CUDA
template<typename Real> inline void cublas_trsm(int m, int n, Real alpha, const Real*\
                                                A, int lda, Real* B, int ldb) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_trsm<float>(int m, int n, float alpha, const float* A, \
                                          int lda, float* B, int ldb) {
  cublasStrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
template<> inline void cublas_trsm<double>(int m, int n, double alpha, const double* \
                                           A, int lda, double* B, int ldb) {
  cublasDtrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
#endif

template<typename Real>
void CuTpMatrix<Real>::Invert() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(this->NumRows(), CUBLOCK));
    CuMatrix<Real> tmp(this->NumRows(), this->NumRows());
    int dim = this->NumRows();
    Real alpha = 1.0;
    cuda_set_diag(dimGrid, dimBlock, tmp.RowData(0), alpha, tmp.Dim());
    //Matrix<Real> A(dim,dim);
    //tmp.CopyToMat(&A);
    cublas_trsm(dim,dim,alpha,this->data_,1,tmp.RowData(0),tmp.Dim().stride);
    this->CopyFromMat(tmp, kNoTrans);
  } else
#endif
  {
    
  }
}

template<typename Real>
void CuTpMatrix<Real>::CopyFromMat(CuMatrixBase<Real> &M,
                                   MatrixTransposeType Trans) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t nr = (this->NumRows())*(this->NumRows() + 1) / 2;
    int dimGrid(1);
    int dimBlock(nr);
    if (Trans == kNoTrans) {
      cuda_take_lower(dimGrid, dimBlock, M.RowData(0), this->data_, M.Dim(), this->NumRows());
      cudaThreadSynchronize();
    } else {
      cuda_take_upper(dimGrid, dimBlock, M.RowData(0), this->data_, M.Dim(), this->NumRows());
      cudaThreadSynchronize();
    }      
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template class CuTpMatrix<float>;
template class CuTpMatrix<double>;

} // namespace
