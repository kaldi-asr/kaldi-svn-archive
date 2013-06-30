// cudamatrix/cuda-test.cc
//
//
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
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
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(          i, j))+std::abs(B(i, j)))));
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
  // CopyFromTp
  TpMatrix<Real> A(10);
  A.SetRandn();
  KALDI_LOG << A;
  CuTpMatrix<Real> B(A);
  CuMatrix<Real> C(B, kTrans);
  Matrix<Real> D(10,10);
  C.CopyToMat(&D);
  KALDI_LOG << D;
  /*
  //TraceSpSp
  for (MatrixIndexT iter = 0; iter < 19; iter++) {
    //int32 dim = 15 + rand() % 10;
    int32 dim = iter + 1;
    KALDI_LOG << "dim is : " << dim << '\n';
    SpMatrix<float> A1(dim);
    A1.SetRandn();
    //A1.SetDiag(1.0);
    //A1(dim-2,dim-2) = 1;
    //A1(dim-1,dim-1) = 10;
    //KALDI_LOG << A1;
    CuSpMatrix<float> A(A1);
    SpMatrix<Real> B1(dim);
    B1.SetRandn();
    //B1.SetDiag(1.0);
    //B1(dim-2,dim-2) = 1;
    //B1(dim-1,dim-1) = 20;
    //KALDI_LOG << B1;
    CuSpMatrix<Real> B(B1);
    KALDI_LOG << TraceSpSp(A1,B1) << '\n';
    KALDI_LOG << TraceSpSp(A,B) << '\n';
  }  
 
  // CopyFromMat
  CuMatrix<float> A(8,10);
  A.SetRandn();
  CuMatrix<Real> B(10,8);
  B.CopyFromMat(A, kTrans);
  CuMatrix<float> C(8,10);
  C.CopyFromMat(B, kTrans);
  Matrix<float> A1(8,10);
  Matrix<Real> B1(10,8);
  Matrix<float> C1(8,10);
  A.CopyToMat(&A1);
  B.CopyToMat(&B1);
  C.CopyToMat(&C1);
  KALDI_LOG << A1;
  KALDI_LOG << B1;
  KALDI_LOG << C1;
  AssertEqual(A1,C1);
  
  A.CopyToMat(&A1);
  //B.CopyToMat(&B1);
  B1.CopyFromMat(A1);
  C1.CopyFromMat(B1);
  KALDI_LOG << A1;
  KALDI_LOG << C1;
  
  //AssertEqual(A1,B1);
  
  CuMatrix<Real> A(12,12);
  A.SetRandn();
  Matrix<Real> B(12,12);
  A.CopyToMat(&B);
  KALDI_LOG << B;
  Real power = 2.0;
  A.ApplyPow(power);
  A.CopyToMat(&B);
  KALDI_LOG << B;
  
  
  CuVector<Real> v(15);
  Vector<Real> v1(15);
  v.SetRandn();
  v.CopyToVec(&v1);
  KALDI_LOG << v1;
  CuMatrix<Real> m(3,15);
  m.CopyRowsFromVec(v);
  Matrix<Real> m1(3,15);
  m.CopyToMat(&m1);
  KALDI_LOG << m1;

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 5 + rand() % 10;
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> A1(dim,dim);
    A1.SetRandn();
    A.CopyFromMat(A1);
    CuVector<Real> C(dim);
    Vector<Real> C1(dim);
    Vector<Real> D(dim);
    //KALDI_LOG << A1;
    for (MatrixIndexT row = 0; row < dim; row++) {
      C.CopyFromVec(A.Row(row));
      C1.CopyFromVec(A1.Row(row));
      C.CopyToVec(&D);
      AssertEqual(C1,D);
    } 
  }


  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 5 + rand() % 10;
    KALDI_LOG << "dim is " << dim << '\n';
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> A1(dim,dim);
    A1.SetRandn();
    A.CopyFromMat(A1);
    for (MatrixIndexT r = 0; r < dim; r++) {
      for (MatrixIndexT c = 0; c < dim; c++)
        std::cout << A(r,c) << ' ';
      std::cout << '\n';
    }
    for (MatrixIndexT r = 0; r < dim; r++) {
      for (MatrixIndexT c = 0; c < dim; c++)
        std::cout << A1(r,c) << ' ';
      std::cout << '\n';
    }
  }
        
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 5 + rand() % 10;
    KALDI_LOG << "dim is " << dim << '\n';
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> A1(dim,dim);
    A1.SetRandn();
    A.CopyFromMat(A1);
    CuMatrix<Real> B(dim,dim);
    Matrix<Real> B1(dim,dim);
    B1.SetRandn();
    B.CopyFromMat(B1);
    Real value = TraceMatMat(A,B,kTrans);
    KALDI_LOG << value << '\n';
    value = TraceMatMat(A1,B1,kTrans);
    KALDI_LOG << value << '\n';
  }
  
 
  SpMatrix<Real> A(dim);
  A.SetRandn();
  
  CuSpMatrix<Real> B(A);
  CuMatrix<Real> C(B);
  Matrix<Real> D(dim,dim);
  C.CopyToMat(&D);
  KALDI_LOG << D;

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

  //CuMatrix<Real> L(dim,dim);
  //L.CopyFromSp(G);
  CuMatrix<Real> L(G);
  Matrix<Real> M(dim,dim);
  L.CopyToMat(&M);
  KALDI_LOG << M << '\n';
  */

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
    B.AddVecVec(1.0, C, C);
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
    KALDI_LOG << "D is: " << D << '\n';
    Matrix<Real> E(dim,dim);
    E.AddMatMat(1.0, D, kNoTrans, D, kTrans, 0.0);
    // check if the D'D is eaual to B or not!
    KALDI_LOG << "B is: " << B << '\n';
    KALDI_LOG << "E is: " << E << '\n';
    AssertEqual(B,E);
  }
}

template<class Real> static void UnitTestTrace() {
  for (MatrixIndexT iter = 1; iter < 18; iter++) {
    MatrixIndexT dim = iter;
    KALDI_LOG << "dim is : " << iter;
    SpMatrix<Real> A(dim);
    A.SetRandn();
    CuSpMatrix<Real> B(A);
    KALDI_LOG << "cpu trace is : " << A.Trace();
    KALDI_LOG << "gpu trace is : " << B.Trace();
  }
  /*
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
  */
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
    X.AddMatMat(1.0, B, kNoTrans, D, kNoTrans, 0.0);
    KALDI_LOG << "X is (should be identity): " << X << '\n';
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

template<class Real> static void UnitTestMatrix() {
  //operator()
  for (MatrixIndexT iter = 0; iter < 2; iter++) {
    int32 dim1 = 6 + rand() % 10;
    int32 dim2 = 8 + rand() % 10;
    Matrix<Real> A(dim1,dim2);
    A.SetRandn();
    CuMatrix<Real> B(A);
    KALDI_LOG << A(0,0) << '\n';
    KALDI_LOG << B(0,0) << '\n';
  }
  //AddMatMatDivMatElements
  for (MatrixIndexT iter = 0; iter < 1; iter++) {
    int32 dim = 6;//15 + rand() % 10;
    CuMatrix<Real> A(dim,dim);
    CuMatrix<Real> B(dim,dim);
    CuMatrix<Real> C(dim,dim);
    CuMatrix<Real> D(dim,dim);
    A.SetRandn();
    B.SetRandn();
    C.SetRandn();
    D.SetRandn();
    Matrix<Real> tmp(dim,dim);
    A.CopyToMat(&tmp);
    KALDI_LOG << tmp;
    B.CopyToMat(&tmp);
    KALDI_LOG << tmp;
    C.CopyToMat(&tmp);
    KALDI_LOG << tmp;
    D.CopyToMat(&tmp);
    KALDI_LOG << tmp;
    A.AddMatMatDivMatElements(1.0,B,kNoTrans,C,kNoTrans,D,kNoTrans,1.0);

    A.CopyToMat(&tmp);
    KALDI_LOG << tmp;
  }
  //SetRandn
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim1 = 15 + rand() % 10;
    int32 dim2 = dim1;//10 + rand() % 14;
    //KALDI_LOG << "dimension is " << dim1
    //          << " " << dim2 << '\n';
    CuMatrix<Real> A(dim1,dim2);
    A.SetRandn();
    Matrix<Real> A1(dim1,dim2);
    A.CopyToMat(&A1);
    //KALDI_LOG << "gpu sum is: " << A.Sum() << '\n';
    //KALDI_LOG << "cpu sum is: " << A1.Sum() << '\n';
  }
}

template<class Real> static void UnitTestVector() {
  // Scale
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 24 + rand() % 10;
    Vector<Real> A(dim);
    A.SetRandn();
    CuVector<Real> B(A);
    Vector<Real> C(dim);
    Real r = 1.43;
    B.Scale(r);
    B.CopyToVec(&C);
    A.Scale(r);
    //KALDI_LOG << A;
    //KALDI_LOG << (A.Scale(r));
    //KALDI_LOG << C;
    AssertEqual(A, C);
  }
  
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + rand() % 10;
    CuVector<Real> A(dim);
    CuVector<Real> B(dim);
    Vector<Real> A1(dim);
    Vector<Real> B1(dim);
    A.SetRandn();
    B.SetRandn();
    A.CopyToVec(&A1);
    B.CopyToVec(&B1);
    A.MulElements(B);
    A1.MulElements(B1);
    Vector<Real> A2(dim);
    A.CopyToVec(&A2);
    AssertEqual(A1,A2);
  }
  /*
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 72;
    CuVector<Real> A(dim);
    Vector<Real> A1(dim);
    CuMatrix<Real> B(9,8);
    Matrix<Real> B1(9,8);
    B.SetRandn();
    B.CopyToMat(&B1);
    A.CopyRowsFromMat(B);
    A1.CopyRowsFromMat(B1);
    Vector<Real> A2(dim);
    A.CopyToVec(&A2);
    AssertEqual(A1,A2);
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + rand() % 10;
    CuVector<Real> A(dim);
    A.SetRandn();
    Vector<Real> A1(dim);
    A.CopyToVec(&A1);
    KALDI_LOG << "cpu min is : " << A1.Min() << '\n';
    KALDI_LOG << "gpu min is : " << A.Min() << '\n';    
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + rand() % 10;
    CuVector<Real> A(dim);
    A.SetRandn();
    Vector<Real> A1(dim);
    A.CopyToVec(&A1);
    CuVector<Real> B(dim);
    B.SetRandn();
    Vector<Real> B1(dim);
    B.CopyToVec(&B1);
    CuVector<Real> C(dim);
    C.SetRandn();
    Vector<Real> C1(dim);
    C.CopyToVec(&C1);
    Real alpha = 2;
    Real beta = 3;
    A.AddVecVec(alpha, B, C, beta);
    A1.AddVecVec(alpha,B1,C1,beta);
    Vector<Real> D(dim);
    A.CopyToVec(&D);
    AssertEqual(D,A1);
  }
  
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim1 = 15 + rand() % 10;
    int32 dim2 = 10 + rand() % 10;
    Matrix<Real> A(dim1,dim2);
    for (MatrixIndexT i = 0; i < dim1; i++) {
      for (MatrixIndexT j = 0; j < dim2; j++)
        A(i,j) = i + 2 * j + 1;
    }
    KALDI_LOG << A;
    CuMatrix<Real> B(dim1,dim2);
    B.CopyFromMat(A);
    CuVector<Real> C(dim1);
    C.SetZero();
    Real alpha = 1;
    Real beta = 1;
    C.AddDiagMat2(alpha, B, kNoTrans, beta);
    Vector<Real> D(dim1);
    C.CopyToVec(&D);
    KALDI_LOG << D << '\n';
    Vector<Real> E(dim1);
    E.AddDiagMat2(alpha, A, kNoTrans, beta);
    KALDI_LOG << E;
    AssertEqual(D,E);
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim1 = 15 + rand() % 10;
    int32 dim2 = 10 + rand() % 10;
    Matrix<Real> A(dim1,dim2);
    for (MatrixIndexT i = 0; i < dim1; i++) {
      for (MatrixIndexT j = 0; j < dim2; j++)
        A(i,j) = i + 2 * j + 1;
    }
    KALDI_LOG << A;
    CuMatrix<Real> B(dim1,dim2);
    B.CopyFromMat(A);
    CuSubVector<Real> C(B,1);
    Vector<Real> D(dim2);
    C.CopyToVec(&D);
    KALDI_LOG << D;
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + rand() % 10;
    CuVector<Real> A(dim);
    A.SetRandn();
    Vector<Real> A1(dim);
    A.CopyToVec(&A1);
    CuVector<Real> B(dim);
    B.SetRandn();
    Vector<Real> B1(dim);
    B.CopyToVec(&B1);
    Real dot = VecVec(A,B);
    KALDI_LOG << "dot product in gpu: " << dot << '\n';
    dot = VecVec(A1,B1);
    KALDI_LOG << "dot product in cpu: " << dot << '\n';    
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + rand() % 10;
    CuVector<Real> A(dim);
    Vector<Real> A1(dim);
    for (MatrixIndexT i = 0; i < dim; i++)
      A1(i) = i;
    A.CopyFromVec(A1);
    KALDI_LOG << A(dim-2) << '\n';
    KALDI_LOG << A1(dim-2) << '\n';
  }
  */
}

template<class Real>
static void CuMatrixUnitTest(bool full_test) {
  UnitTestSimpleTest<Real>();
  UnitTestTrace<Real>();
  UnitTestCholesky<Real>();
  UnitTestInvert<Real>();
  UnitInvert<Real>();
  UnitTestCopyFromMat<Real>();
  UnitTestCopySp<Real>();
  UnitTestConstructor<Real>();
  UnitTestVector<Real>();
  UnitTestMatrix<Real>();
}
} //namespace

int main() {
  using namespace kaldi;
  kaldi::int32 use_gpu_id = -2; // -2 means automatic selection.
#if HAVE_CUDA == 1
  CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif
  
  bool full_test = false;
  kaldi::CuMatrixUnitTest<double>(full_test);
  KALDI_LOG << "Tests succeeded.\n";
  return 0;
}
