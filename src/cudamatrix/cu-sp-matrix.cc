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
