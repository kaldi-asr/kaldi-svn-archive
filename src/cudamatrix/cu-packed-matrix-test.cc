// cudamatrix/cu-sp-matrix-test.cc
//
//
// UnitTests for testing cu-sp-matrix.h methods.
//

#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "cu-device.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-math.h"

using namespace kaldi;

namespace kaldi {

/*
 * INITIALIZERS
 */

/*
 * ASSERTS
 */
template<class Real>
static void AssertEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0; i < A.Dim(); i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}

template<class Real>
static bool ApproxEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0; i < A.Dim(); i++)
    if (std::abs(A(i)-B(i)) > tol) return false;
  return true;
}

static void AssertEqual(std::vector<int32> &A, std::vector<int32> &B) {
  KALDI_ASSERT(A.size() == B.size());
  for (size_t i = 0; i < A.size(); i++)
    KALDI_ASSERT(A[i] == B[i]);
}

template<class Real>
static void AssertEqual(const CuPackedMatrix<Real> &A,
                        const CuPackedMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}

template<class Real>
static void AssertEqual(const PackedMatrix<Real> &A,
                        const PackedMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}

template<class Real>
static void AssertEqual(const PackedMatrix<Real> &A,
                        const CuPackedMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}

template<class Real>
static bool ApproxEqual(const PackedMatrix<Real> &A,
                        const PackedMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  PackedMatrix<Real> diff(A);
  diff.AddPacked(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min()),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

/*
 * Unit Tests
 */
template<class Real>
static void UnitTestCuPackedMatrixConstructor() { 
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;

    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);
    CuPackedMatrix<Real> C(B);
    AssertEqual(B, C);
  }
}

template<class Real>
static void UnitTestCuPackedMatrixCopy() { 
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    CuPackedMatrix<Real> C(dim);
    C.CopyFromPacked(A);
    CuPackedMatrix<Real> D(dim);
    D.CopyFromPacked(B);
    AssertEqual(C, D);

    PackedMatrix<Real> E(dim);
    D.CopyToPacked(&E);
    AssertEqual(A, E);
  }
}

template<class Real>
static void UnitTestCuPackedMatrixTrace() {
  for (MatrixIndexT i = 1; i < 50; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);
    
#ifdef KALDI_PARANOID
    KALDI_ASSERT(A.Trace(), B.Trace());
#endif
  }
}

template<class Real>
static void UnitTestCuPackedMatrixScale() {
  for (MatrixIndexT i = 1; i < 50; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    Real scale_factor = 23.5896223;
    A.Scale(scale_factor); 
    B.Scale(scale_factor);
    AssertEqual(A, B);
  }
}

template<class Real>
static void UnitTestCuPackedMatrixScaleDiag() {
  for (MatrixIndexT i = 1; i < 50; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    Real scale_factor = 23.5896223;
    A.ScaleDiag(scale_factor); 
    B.ScaleDiag(scale_factor);
    AssertEqual(A, B);
  }
}

template<class Real> void CudaPackedMatrixUnitTest() {
  UnitTestCuPackedMatrixConstructor<Real>();
  UnitTestCuPackedMatrixCopy<Real>();
  UnitTestCuPackedMatrixTrace<Real>();
  UnitTestCuPackedMatrixScale<Real>();
}

} // namespace kaldi


int main() {
  using namespace kaldi;
  // Select the GPU
  kaldi::int32 use_gpu_id = -2;
#if HAVE_CUDA == 1
  CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif
  kaldi::CudaPackedMatrixUnitTest<float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CudaPackedMatrixUnitTest<double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CudaPackedMatrixUnitTest<double>();
#endif
  
  KALDI_LOG << "Tests succeeded";
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}
