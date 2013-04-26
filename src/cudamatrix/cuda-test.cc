#include <iostream>
#include "base/kaldi-common.h"
#include <ctime>

#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-vector.h"
#include <numeric>
#include <time.h>

namespace kaldi {

/*
 * INITIALIZERS
 */ 
template<class Real>
static void InitRand(SpMatrix<Real> *M) {
  do {
    for (MatrixIndexT i = 0; i < M->NumRows(); i++) {
      for (MatrixIndexT j = 0; j <= i; j++ ) {
	(*M)(i,j) = RandGauss();
      }
    }
  } while (M->NumRows() != 0 && M->Cond() > 100);
}

template<class Real>
static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0; i < v->Dim(); i++) {
    (*v)(i) = RandGauss();
  }
}
/*
 * ASSERTS
 */
template<class Real>
static void AssertEqual(const VectorBase<Real> &A,
                        const VectorBase<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0; i < A.Dim(); i++) {
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol * std::max(1.0, (double) (std::abs(A(i)) + std::abs(B(i)))));
  }
}

template<class Real>
static void AssertEqual(const MatrixBase<Real> &A,
                        const MatrixBase<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(\
          i, j))+std::abs(B(i, j)))));
    }
  }
}

template<class Real>
static void AssertEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j <= i;j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(\
          i, j))+std::abs(B(i, j)))));
    }
  }
}

template<class Real> 
static void ApproxEqual(const SpMatrix<Real> &A,
			const SpMatrix<Real> &B,
			float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
}

// ApproxEqual
template<class Real>
static bool ApproxEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  SpMatrix<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
    d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

template<class Real>
static void UnitTestSimpleTest() {
  // test differnet constructors + CopyFrom* + CopyToMat
  int32 dim = 5 + rand() % 10;
  std::cout << "dim is : " << dim << std::endl;
  CuPackedMatrix<Real> A(dim);
  PackedMatrix<Real> B(dim);
  A.CopyToMat(&B);
  std::cout << "The cudamatrix A is" << std::endl;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << B(i,j) << " ";
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < dim; i++) {
    B(i,i) = i;
  }
  CuPackedMatrix<Real> C(B);
  PackedMatrix<Real> D(dim);
  C.CopyToMat(&D);
  std::cout << "The cudamatrix C is" << std::endl;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << D(i,j) << " ";
      D(i,j) = D(i,j) + 1;
    }
    std::cout << std::endl;
  }
  C.CopyFromPacked(D);
  CuPackedMatrix<Real> E(C);
  PackedMatrix<Real> F(dim);
  E.CopyToMat(&F);
  std::cout << "The cudamatrix E is" << std::endl;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << F(i,j) << " ";
    }
    std::cout << std::endl;
  }
  E.SetDiag(10);
  E.Scale(2);
  E.ScaleDiag(4);
  std::cout << "Trace(E) = " << E.Trace() << std::endl;
  
  
  CuSpMatrix<Real> G(dim);
  G.SetDiag(10);
  G.Invert();
  SpMatrix<Real> H(dim);
  KALDI_LOG << "NUMROWS is" << H.NumRows() << '\n';
  G.CopyToMat(&H);
  H(1,1)=14;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j <= i; j++) {
      std::cout << H(i,j) << " ";
    }
    std::cout << std::endl;
  }

  Vector<Real> I(dim);
  InitRand(&I);
  CuVector<Real> J(dim);
  J.CopyFromVec(I);

  G.AddVec2(1,J);
  SpMatrix<Real> K(dim);
  G.CopyToMat(&K);

  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++) {
      std::cout << K(i,j) << " ";
    }
    std::cout << std::endl;
  }

  CuMatrix<Real> L(dim,dim);
  L.CopyFromSp(G);
  Matrix<Real> M(dim,dim);
  L.CopyToMat(&M);
  KALDI_LOG << M << '\n';

}

template<class Real> static void UnitTestCholesky() {
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    MatrixIndexT dim = 45 + rand() %  40;
    // set dimension
    //MatrixIndexT dim = 13;
    // computing the matrix for cholesky input
    // CuMatrix is cuda matrix class while Matrix is cpu matrix class
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> B(dim,dim);
    Vector<Real> C(dim);
    for (MatrixIndexT i = 0; i < dim; i++) {
      B(i,i) = 1;
      C(i) = i + 1;
    }
    B.AddVecVec(1.0,C,C);
    // copy the matrix to cudamatrix object
    A.CopyFromMat(B);
    A.CopyToMat(&B);
    KALDI_LOG << B << '\n';
    // doing cholesky
    A.Cholesky();
    Matrix<Real> D(dim,dim);
    A.CopyToMat(&D);
    for (MatrixIndexT i = 0; i < dim; i++) {
      for (MatrixIndexT j = i+1; j < dim; j++)
        D(i,j) = 0;
    }
    KALDI_LOG << D << '\n';
    Matrix<Real> E(dim,dim);
    E.AddMatMat(1,D,kNoTrans,D,kTrans,0);
    // check if the D'D is eaual to B or not!
    AssertEqual(B,E);
  }
}

template<class Real> static void UnitTestTrace() {
  Vector<Real> tim(100);
  Vector<Real> d(100);
  for (MatrixIndexT iter = 0; iter < 100; iter++) {
    MatrixIndexT dim = 10000 + rand() % 400;
    Matrix<Real> A(dim,dim);
    A.SetRandn();
    CuMatrix<Real> B(A);
    CuSpMatrix<Real> C(B,kTakeLower);
    clock_t t1 = clock();
    tim(iter) = C.Trace();
    clock_t t2 = clock();
    //tim(iter) = t2 - t1;
    d(iter) = dim;
    KALDI_LOG << tim(iter) << iter << '\n';
    KALDI_LOG << d(iter) << iter << '\n';
  }
  KALDI_LOG << "tim is " << tim << '\n';
  KALDI_LOG << "dim is " << d << '\n';
}

template<class Real> static void UnitInvert() {
  //MatrixIndexT dim = 15 + rand() %  40;;
  MatrixIndexT dim = 8;
  CuMatrix<Real> A(dim,dim);
  Matrix<Real> B(dim,dim);
  Vector<Real> C(dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    B(i,i) = 1;
    C(i) = i + 1;
  }
  B.AddVecVec(1.0,C,C);
  CuMatrix<Real> tmp(dim,dim);
  A.CopyFromMat(B);
  //A.Cholesky();
  A.CopyToMat(&B);
  KALDI_LOG << "B is : " << '\n';
  KALDI_LOG << B << '\n';
  A.Invert(1.0, tmp);
  Matrix<Real> D(dim,dim);
  A.CopyToMat(&D);
  KALDI_LOG << "D is : " << '\n';
  KALDI_LOG << D << '\n';
  Matrix<Real> X(dim,dim);
  X.AddMatMat(1,B,kNoTrans,D,kNoTrans,0);
  KALDI_LOG << X << '\n';
  //for (MatrixIndexT i = 0; i < dim; i++) {
  //  for (MatrixIndexT j = i+1; j < dim; j++)
  //    D(i,j) = 0;
  //}
  //Matrix<Real> E(dim,dim);
  //E.AddMatMat(1,D,kNoTrans,D,kTrans,0);
  //AssertEqual(B,E);
}

template<class Real> static void UnitTestInvert() {
  for (MatrixIndexT iter = 0; iter < 1; iter++) {
    // MatrixIndexT dim = 15 + rand() % 40;
    MatrixIndexT dim = 50;
    KALDI_LOG << "dim is : " << '\n';
    KALDI_LOG << dim << '\n';
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> B(dim,dim);
    Vector<Real> C(dim);
    for (MatrixIndexT i = 0; i < dim; i++) {
      B(i,i) = 1;
      C(i) = i + 1;
    }
    Matrix<Real> Identity(B);
    B.AddVecVec(1.0,C,C);
    CuMatrix<Real> tmp(dim,dim);
    A.CopyFromMat(B);
    KALDI_LOG << "B is " << '\n';
    KALDI_LOG << B << '\n';
    
    A.Invert(1.0, tmp);
    Matrix<Real> D(dim,dim);
    A.CopyToMat(&D);
    KALDI_LOG << "D is " << '\n';
    KALDI_LOG << D << '\n';
    Matrix<Real> X(dim,dim);
    X.AddMatMat(1,B,kNoTrans,D,kNoTrans,0);
    KALDI_LOG << X << '\n';
    AssertEqual(Identity,X);
  }
}

template<class Real> static void UnitTestConstructor() {
  MatrixIndexT dim = 8;
  CuMatrix<Real> A(dim,dim);
  Matrix<Real> B(dim,dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <=i; j++)
      B(i,j) = i+j;
    for (MatrixIndexT j = i+1; j < dim; j++)
      B(i,j) = i+j+4;
  }
  KALDI_LOG << "A is : " << '\n';
  KALDI_LOG << B << '\n';
  A.CopyFromMat(B);
  //CuSpMatrix<Real> C(dim);
  //C.CopyFromMat(A,kTakeLower);
  CuSpMatrix<Real> C(A, kTakeLower);
  SpMatrix<Real> D(dim);
  C.CopyToMat(&D);
  KALDI_LOG << "C is : " << '\n';
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }  
}

template<class Real> static void UnitTestCopySp() {
  // Checking that the various versions of copying                                 
  // matrix to SpMatrix work the same in the symmetric case.                         
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    int32 dim = 5 + rand() %  10;
    SpMatrix<Real> A(dim), B(dim);
    A.SetRandn();
    Matrix<Real> C(A);
    //CuMatrix<Real> D(C);
    
    {
      CuMatrix<Real> D2(dim,dim);
      D2.CopyFromMat(C);
      KALDI_LOG << "D2 is " << D2;
      CuSpMatrix<Real> E(D2.NumRows(), kUndefined);
      KALDI_LOG << "D2 is " << D2;
      E.CopyFromMat(D2, kTakeLower);
      KALDI_LOG << "D2 is " << D2;
    }
    
    CuMatrix<Real> D(dim,dim);
    D.CopyFromMat(C);
    KALDI_LOG << "D stride is : " << D.Stride() <<'\n';
    
    CuSpMatrix<Real> E(D,kTakeLower);
    ///CuSpMatrix<Real> E(dim);
    //E.CopyFromMat(D,kTakeLower);
    /*
    KALDI_LOG << D.NumRows() << '\n';
    //E.CopyFromMat(D, kTakeMean);
    //E(D, kTakeMean);
    //KALDI_LOG << E.NumRows() << '\n';
    /*
    E.CopyToMat(&B);
    AssertEqual(A, B);
    B.SetZero();
    //E.CopyFromMat(D, kTakeLower);
    CuSpMatrix<Real> F(D,kTakeLower);
    //F(D, kTakeLower);
    F.CopyToMat(&B);
    AssertEqual(A, B);
    B.SetZero();
    //E.CopyFromMat(D, kTakeUpper);
    //E(D, kTakeUpper);
    CuSpMatrix<Real> G(D, kTakeUpper);
    G.CopyToMat(&B);
    AssertEqual(A, B);
    */  
  }
  
}

template<class Real> static void UnitTestCopyFromMat() {
  MatrixIndexT dim = 8;
  CuMatrix<Real> A(dim,dim);
  Matrix<Real> B(dim,dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <=i; j++)
      B(i,j) = i+j;
    for (MatrixIndexT j = i+1; j < dim; j++)
      B(i,j) = i+j+4;
  }
  KALDI_LOG << "A is : " << '\n';
  KALDI_LOG << B << '\n';
  A.CopyFromMat(B);
  CuSpMatrix<Real> C(dim);
  C.CopyFromMat(A,kTakeLower);
  SpMatrix<Real> D(dim);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : " << '\n';
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }
  C.CopyFromMat(A,kTakeUpper);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : " << '\n';
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }
  
  C.CopyFromMat(A,kTakeMean);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : " << '\n';
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }
  
  //KALDI_LOG << D << '\n';
}
template<class Real> static void UnitTestVector() {
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + rand() % 10;
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> B(dim,dim);
    B.SetRandn();
    A.CopyFromMat(B);
    KALDI_LOG << B << '\n';
    CuVector<Real> C(dim);
    C.CopyColFromMat(A,1);
    Vector<Real> D(dim);
    C.CopyToVec(&D);    
    KALDI_LOG << D << '\n';
    C.CopyColFromMat(A,dim-2);
    C.CopyToVec(&D);    
    KALDI_LOG << D << '\n';
    /*
    KALDI_LOG << iter << '\n';
    int32 dim = 5 + rand() % 10;
    Vector<Real> A(dim);
    InitRand(&A);
    //for (MatrixIndexT i = 0; i < dim; i++)
    //  A(i) = i+1;
    CuVector<Real> B(dim);
    B.CopyFromVec(A);
    Vector<Real> C(dim);
    B.CopyToVec(&C);
    //AssertEqual(A,C);
    KALDI_LOG << A.Sum() << '\n';
    KALDI_LOG << B.Sum() << '\n';
    */
  }

}

template<class Real>
static void CuMatrixUnitTest(bool full_test) {
  //UnitTestSimpleTest<Real>();
  //UnitTestTrace<Real>();
  //UnitTestCholesky<Real>();
  //UnitTestInvert<Real>();
  //UnitInvert<Real>();
  //UnitTestCopyFromMat<Real>();
  //UnitTestCopySp<Real>();
  //UnitTestConstructor<Real>();
  UnitTestVector<Real>();
}
} //namespace

int main() {
  using namespace kaldi;
  bool full_test = false;
  kaldi::CuMatrixUnitTest<double>(full_test);
  KALDI_LOG << "Tests succeeded.\n";
  return 0;
}
