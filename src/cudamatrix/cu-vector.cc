#if HAVE_CUDA==1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-randkernels.h"
#include "cu-math.h"
#include "cu-vector.h"
#include "cu-matrix.h"
#include "cu-rand.h"

namespace kaldi {


template<typename Real>
void CuVectorBase<Real>::CopyColFromMat(const CuMatrixBase<Real> &mat, MatrixIndexT col) {
  KALDI_ASSERT(col < mat.NumCols());
  KALDI_ASSERT(dim_ == mat.NumRows());
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {

    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));

    cuda_copy_col_from_mat(dimGrid, dimBlock, data_, col, mat.RowData(0), mat.Dim(), dim_);
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyColFromMat", tim.Elapsed());
  } else
#endif
  {
    Vec().CopyColFromMat(mat.Mat(),col);
  }
}


template<typename Real>
void CuVectorBase<Real>::SetRandn() {
  CuRand<Real> tmp;
  tmp.RandGaussian(this);
}


template<typename Real>
Real CuVectorBase<Real>::Sum() {
  Real sum_value = 0;
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {

    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));

    Real* device_sum_value;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_sum_value), sizeof(Real)))
    CU_SAFE_CALL(cudaMemset(device_sum_value,0, sizeof(Real)));
    cuda_vec_sum(dimGrid, dimBlock, data_, device_sum_value, dim_);
    CU_SAFE_CALL(cudaMemcpy(&sum_value, device_sum_value, sizeof(Real), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuVectorBase::Sum", tim.Elapsed());
  } else
#endif
  {
    sum_value = Vec().Sum();
  }
  return sum_value;
}


template<typename Real>
MatrixIndexT CuVectorBase<Real>::ApplyFloor(Real floor_val) {
  MatrixIndexT num_floored = 0;
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {

    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));

    int* device_num_floored;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_num_floored), sizeof(int)))
    CU_SAFE_CALL(cudaMemset(device_num_floored,0, sizeof(int)));
    cuda_vec_apply_floor(dimGrid, dimBlock, data_, floor_val, device_num_floored, dim_);
    CU_SAFE_CALL(cudaMemcpy(&num_floored, device_num_floored, sizeof(int), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyFloor", tim.Elapsed());
  } else
#endif
  {
    num_floored = Vec().ApplyFloor(floor_val);
  }
  return num_floored;

}


template<typename Real>
void CuVectorBase<Real>::ApplyExp() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {

    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));

    cuda_vec_apply_exp(dimGrid, dimBlock, data_, dim_);
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyExp", tim.Elapsed());
  } else
#endif
  {
    Vec().ApplyExp();
  }
}


template<typename Real>
void CuVectorBase<Real>::ApplyLog() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {

    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));

    Real* device_flag;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_flag), sizeof(Real)));
    CU_SAFE_CALL(cudaMemset(device_flag, 0, sizeof(Real)));
    cuda_vec_apply_log(dimGrid, dimBlock, data_, device_flag, dim_);
    Real host_flag = 0.0;
    CU_SAFE_CALL(cudaMemcpy(&host_flag, device_flag, sizeof(Real), cudaMemcpyDeviceToHost));
    if (host_flag > 0)
      KALDI_ERR << "Trying to take log of a negative number.";
    
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyLog", tim.Elapsed());

  } else
#endif
  {
    Vec().ApplyLog();
  }
}

#if HAVE_CUDA==1
template<typename Real> inline void cublas_gemv(char trans, int m, int n, Real alpha,
                                                const Real* A, int lda, const Real* x,
                                                int incx, Real beta, Real* y, int incy) {
  KALDI_ERR << __func__ << " Not implemented! ";
}
template<> inline void cublas_gemv(char trans, int m, int n, float alpha,
                                   const float* A, int lda, const float* x,
                                   int incx, float beta, float* y, int incy) {
  cublasSgemv(trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}
template<> inline void cublas_gemv(char trans, int m, int n, double alpha,
                                   const double* A, int lda, const double* x,
                                   int incx, double beta, double* y, int incy) {
  cublasDgemv(trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}
#endif

template<typename Real>
void CuVectorBase<Real>::AddMatVec(const Real alpha,
                                   const CuMatrixBase<Real> &M,
                                   MatrixTransposeType trans,
                                   const CuVectorBase<Real> &v,
                                   const Real beta) {
  KALDI_ASSERT((trans == kNoTrans && M.NumCols() == v.dim_ && M.NumRows() == dim_) || (trans == kTrans && M.NumRows() == v.dim_ && M.NumCols() == dim_));
  KALDI_ASSERT(&v != this);
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
  cublas_gemv((trans==kTrans?'T':'N'), M.NumRows(), M.NumCols(), alpha, M.RowData(0), M.Stride(), v.Data(), 1, beta, data_, 1);

  CU_SAFE_CALL(cublasGetError());
  CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().AddMatVec(alpha,M.Mat(),trans,v.Vec(),beta);
  }
}

template<typename Real>
void CuVectorBase<Real>::AddVecVec(Real alpha, const CuVectorBase<Real> &v,
                                   const CuVectorBase<Real> &r, Real beta) {
  KALDI_ASSERT((dim_ == v.dim_ && dim_ == r.dim_));
  KALDI_ASSERT(this != &v && this != &r);
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {

    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));

    cuda_add_vec_vec(dimGrid, dimBlock, alpha, data_, v.Data(), r.Data(), beta, dim_);
    CuDevice::Instantiate().AccuProfile("CuVectorBase::AddVecVec", tim.Elapsed());
  } else
#endif
  {
    KALDI_LOG << "Salam!" << '\n';
    Vec().AddVecVec(alpha, v.Vec(), r.Vec(), beta);
  }
}


template<typename Real>
void CuVectorBase<Real>::AddDiagMat2(Real alpha, const CuMatrixBase<Real> &M,
                                     MatrixTransposeType trans, Real beta) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CUBLOCK);
    int dimGrid(n_blocks(dim_,CUBLOCK));
    if (trans == kNoTrans) {
      cuda_add_diag_mat(dimGrid, dimBlock, alpha, data_, M.RowData(0), beta, M.Dim(), dim_);
    } else {
      cuda_add_diag_mat_trans(dimGrid, dimBlock, alpha, data_, M.RowData(0), beta, M.Dim(), dim_);
    }
    CuDevice::Instantiate().AccuProfile("CuVectorBase::AddDiagMat2", tim.Elapsed());
  } else
#endif
  {
    Vec().AddDiagMat2(alpha, M.Mat(), trans, beta);
  }
  
}


template class CuVectorBase<float>;
template class CuVectorBase<double>;

} // namespace
