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
#include "cu-matrix.h"

namespace kaldi {


template<typename Real>
void CuSpMatrix<Real>::CopyFromMat(const CuMatrixBase<Real> &M,
                                   SpCopyType copy_type) {
  KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT D = this->NumRows();
    dim3 dimBlock(CUBLOCK,CUBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(),CUBLOCK), n_blocks(M.NumRows(),CUBLOCK));
    switch (copy_type) {
      case kTakeMeanAndCheck:
        KALDI_LOG << "kTakeMeanAndCheck!" << '/n';
      case kTakeMean:
        {
          cuda_take_mean(dimGrid, dimBlock, M.RowData(0), this->data_, M.Dim(), D);
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      case kTakeLower:
        {
          cuda_take_lower(dimGrid, dimBlock, M.RowData(0), this->data_, M.Dim(), D);
          cudaThreadSynchronize();
        }
        break;
      case kTakeUpper:
        {
          cuda_take_upper(dimGrid, dimBlock, M.RowData(0), this->data_, M.Dim(), D);
        }
        break;
      default:
        KALDI_ASSERT("Invalid argument to CuSpMatrix::CopyFromMat");
    }
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::Invert", tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), kTakeLower);
  }
}


template<class Real>
void CuSpMatrix<Real>::Invert(Real* logdet, Real* det_sign,
                              bool inverse_needed) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    SpMatrix<Real> mat(this->num_rows_);
    this->CopyToSp(&mat);
    mat.Invert();
    CopyFromSp(mat);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::Invert", tim.Elapsed());
  } else
#endif
  {
    Mat().Invert(logdet, det_sign, inverse_needed);
  }
}

template<class Real>
void CuSpMatrix<Real>::AddVec2(const Real alpha, const CuVectorBase<Real> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t nr = this->num_rows_;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(nr, CUBLOCK), n_blocks(nr, CUBLOCK));

    Real* data = this->data_;
    const Real* v_data = v.Data();

    cuda_add_vec2(dimGrid, dimBlock, data, v_data, alpha, nr);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddVec2", tim.Elapsed());
  } else
#endif
  {
    Mat().AddVec2(alpha, v.Vec());
  }
}

#if HAVE_CUDA==1
template<typename Real> inline void cublas_syrk(char uplo, char trans, int n, int k,
                                                Real alpha, const Real *A, int lda,
                                                Real beta, Real *C, int ldc) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_syrk(char uplo, char trans, int n, int k,
                                    float alpha, const float *A, int lda,
                                    float beta, float *C, int ldc) {
  cublasSsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
template<> inline void cublas_syrk(char uplo, char trans, int n, int k,
                                   double alpha, const double *A, int lda,
                                   double beta, double *C, int ldc) {
  cublasDsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
#endif

template<class Real>
void CuSpMatrix<Real>::AddMat2(const Real alpha, const CuMatrix<Real> &M,
                               MatrixTransposeType transM, const Real beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
               || (transM == kTrans && this->NumRows() == M.NumCols()));

#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT this_dim = this->NumRows(),
        m_other_dim = (transM == kNoTrans ? M.NumCols() : M.NumRows());

    if (this_dim == 0) return;
    if (alpha == 0.0) {
      if (beta != 1.0) this->Scale(beta);
      return;
    }

    //CuMatrix<Real> tmp_mat(*this);
    cublas_syrk('U', transM, this_dim, m_other_dim, alpha, M.RowData(0),
                M.Stride(), beta, this->Data(), 1);
    //this->CopyFromMat(tmp_mat, kTakeLower);
    
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddVEc2", tim.Elapsed());
  } else
#endif
  {
    Mat().AddMat2(alpha, M.Mat(), transM, beta);
  }
}

/*
#if HAVE_CUDA==1
template<typename Real> inline void cublas_trsm(int m, int n, Real alpha,
                                                const Real *A, int lda, Real *B,
                                                int ldb) { 
  KALDI_ERR << __func__ << " Not implemented!"; 
}
template<> inline void cublas_trsm<float>(int m, int n, float alpha,
                                          const float *A, int lda, float *B,
                                          int ldb) {
  cublasStrsm('L', 'U', 'N', 'N', m, n, alpha, A, lda, B, ldb);
}
template<> inline void cublas_trsm<double>(int m, int n, float alpha,
                                           const float *A, int lda, float *B,
                                           int ldb) {
  cublasDtrsm('L', 'U', 'N', 'N', m, n, alpha, A, lda, B, ldb);
}
#endif
*/
template class CuSpMatrix<float>;
template class CuSpMatrix<double>;

} // namespace
