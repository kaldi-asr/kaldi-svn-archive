#include "character-matrix.h"
#include <iostream>

// hhx

namespace kaldi {

template<class Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.003) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j < A.NumCols();j++) {
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol);
    }
}
//
template <typename Real>
static void  GenerateMatrix4U (Matrix<Real> &M) {
  int32 num_rows = M.NumRows(), 
        num_cols = M.NumCols();
  srand ( time(NULL) );
  Real min = static_cast<Real>(rand() % 10);
  KALDI_ASSERT(num_rows > 0 && num_cols > 0);
  std::vector<Real> v(256);
  for(size_t i = 0; i < v.size(); i++) {
    v[i] = static_cast<Real>(i*1.0/255);
  }
  for(int32 row = 0; row < num_rows; ++row)
    for(int32 col = 0; col < num_cols; ++col) {
      int32  k = rand() % 256;
      M(row, col) = min + v[k];
    }
}
//
template <typename Real>
static void  GenerateMatrix4S (Matrix<Real> &M) {
   int32 num_rows = M.NumRows(), 
         num_cols = M.NumCols();
   KALDI_ASSERT(num_rows > 0 && num_cols > 0);
   srand(time(NULL));
   Real base = static_cast<Real>(rand()%20);
   std::vector<Real> v(256);
   for(int32 i = -128, j=0; i < 128; i++, j++) {
     Real x;
     if(i <= 0) {
       x = static_cast<Real>(i*1.0/128);
     } else {
       x = static_cast<Real>(i*1.0/127);
     }
     v[j] = x;
   }
   for(int32 row = 0; row < num_rows; ++row)
    for(int32 col = 0; col < num_cols; ++col) {
      int32  k = rand() % 256;
      M(row, col) = base + v[k];
    }
}
// use gdb to display these
template<typename Real>
static void ShowMatrix(const Matrix<Real> &M) {
  std::cout << "\nrow=" << M.NumRows()
            << ", col=" << M.NumCols()
            << "\nmax=" << M.Max() 
            <<", min=" <<M.Min() << "\n";
  for(int32 row = 0; row < M.NumRows(); ++row) {
    for(int32 col = 0; col < M.NumCols(); ++col) {
      std::cout << M(row,col) << " ";
    }
    std::cout << "\n";
  }
}
//
template<typename Real>
static void TestCopyMat01() {
  int32  row = 7 + rand() % 5;
  int32  col = 15 + rand() % 7;
  Matrix<Real> M(row, col);
  GenerateMatrix4S(M);
  CharacterMatrix<signed char> C;
  C.CopyFromMat(M);
  Matrix<Real> M1;
  C.RecoverMatrix(M1);
  ShowMatrix(M);
  ShowMatrix(M1);
  AssertEqual(M,M1,0.0079);
  // for unsigned char
  Matrix<Real> M2(row, col);
  GenerateMatrix4U(M2);
  CharacterMatrix<unsigned char> C2;
  C2.CopyFromMat(M2);
  Matrix<Real> M3;
  C2.RecoverMatrix(M3);
  AssertEqual(M2,M3,0.004); 

}
// 
template<typename Real> 
static void TestCopyMat02() {
  int32 num_rows = 2;
  int32 num_cols = 10;
  Matrix<Real> M(num_rows,num_cols);
  Real res = static_cast<Real>(1.0/ 10);
  Real base;
  for(int32 i = 0; i < M.NumRows(); i++) {
     base = 0;
    for(int32 j = 0; j < M.NumCols(); j++) {
      M(i,j) = base + res;
      base += res;
    }
  }
  CharacterMatrix<unsigned char> C;
  C.CopyFromMat(M);
  Matrix<Real> M1;
  C.RecoverMatrix(M1);
  for(int32 i = 0; i < M1.NumRows() ; i++) {
    for(int32 j = 0; j < M1.NumCols(); j++) {
      Real x = M1(i,j);
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  AssertEqual(M,M1);
}

template<typename Real>
static void TestAddMatMat () {
  int32  row = 7 + rand() % 5;
  int32  col = 15 + rand() % 7;
  Matrix<Real> M1(row, col);
  GenerateMatrix4U(M1);
  
  int32 row2 = 15 + rand() % 4;
  Matrix<Real> M2(row2,col);
  GenerateMatrix4S(M2);
  
  Matrix<Real> M(row,row2);
  M.AddMatMat(1.0, M1, kNoTrans, M2, kTrans, 0);
  M.ShowMatrix(M);
  // I am writing it up
}


} 

int main() {
  kaldi::TestCopyMat01<float>();
  kaldi::TestCopyMat01<double>();
  kaldi::TestAddMatMat<float>();
  kaldi::TestAddMatMat<double>();
  KALDI_LOG << "character-matrix-test succeeded.\n";
  return 0;
}
