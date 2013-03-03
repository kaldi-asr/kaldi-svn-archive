 #include <iostream>
#include "base/kaldi-common.h"


#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include <numeric>
#include <time.h>

namespace kaldi {

template<class Real>
static void SimpleTest() {
  int32 dim = 5 + rand() % 10;
  std::cout << "dim is : " << dim << std::endl;
  //CuPackedMatrix<Real> S;
  CuPackedMatrix<Real> S(dim);
  PackedMatrix<Real> T(dim);
  S.CopyToMat(&T);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << T(i,j) << " ";
    }
    std::cout << std::endl;
  }
  T(0,0) = 10;
  CuPackedMatrix<Real> U(dim);
  U.CopyFromPacked(T);
  //std::cout << U(0,0) << std::endl;
  /*  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << U(i,j) << " ";
    }
    std::cout << std::endl;
    }*/


  CuSpMatrix<Real> Sp(dim);
  std::cout << Sp.NumRows() << std::endl;
  std::cout << Sp.NumCols() << std::endl;

  // Sp test 
  /*
  SpMatrix<Real> Sp2(dim);
  Sp.CopyToMat(Sp2);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << Sp2(i,j) << " ";
    }
    std::cout << std::endl;
  }
  
  Sp2(0,0) = 10;
  
  CuSpMatrix<Real> CuSp(&Sp2);
  //InitRand(&Sp);
*/
}

/*
 * INITIALIZERS
 */ 
template<class Real>
static void InitRand(MatrixBase<Real> *M) {
  do {
    for (MatrixIndexT i = 0; i < M->NumRows(); i++) {
      for (MatrixIndexT j = 0; j <= i; j++ ) {
	(*M)(i,j) = RandGauss();
      }
    }
  } while (M->NumRows() != 0 && M->Cond() > 100);
}

// Initialization
template<class Real> static void InitRand(CuSpMatrix<Real> *M) {
 start:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
    for (MatrixIndexT j = 0;j<=i;j++)
      (*M)(i, j) = RandGauss();
  if (M->NumRows() != 0 && M->Cond() > 100)
    goto start;
}

// ApproxEqual
template<class Real>
static bool ApproxEqual(const CuSpMatrix<Real> &A,
                        const CuSpMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  CuSpMatrix<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
    d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

template<class Real> static void AssertEqual(const CuSpMatrix<Real> &A,
					     const CuSpMatrix<Real> &B,
					     float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
    for (MatrixIndexT j = 0;j<=i;j++)
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
}

template<class Real> static void UnitTestCopySp() {
  // Checking that the various versions of copying                                         
  // matrix to SpMatrix work the same in the symmetric case.                               
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    int32 dim = 5 + rand() %  10;
    CuSpMatrix<Real> S(dim), T(dim);
    S.SetRandn();
    Matrix<Real> M(S);
    T.CopyFromMat(M, kTakeMeanAndCheck);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeMean);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeLower);
    AssertEqual(S, T);
    T.SetZero();
    T.CopyFromMat(M, kTakeUpper);
    AssertEqual(S, T);
  }
}

template<class Real>
static void CuMatrixUnitTest(bool full_test) {
  SimpleTest<Real>();
}
} //namespace

int main() {
  using namespace kaldi;
  bool full_test = false;
  kaldi::CuMatrixUnitTest<double>(full_test);
  KALDI_LOG << "Tests succeeded.\n";
  return 0;
}
