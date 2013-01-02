#include "character-matrix.h"
#include <iostream>
#include <stdio.h>
#include <stdint.h>
 
 namespace kaldi {
 typedef int32_t MatrixIndexT;
 template<typename Real> 
 static void TestCopyMat() {
   MatrixIndexT num_rows = 2;
   MatrixIndexT num_cols = 10;
   CharacterMatrix<Real> M(num_rows,num_cols);
   Real res = static_cast<Real>(1.0/ 10);
   Real base;
   for(MatrixIndexT i = 0; i < M.NumRows(); i++) {
      base = 0;
     for(MatrixIndexT j = 0; j < M.NumCols(); j++) {
       M(i,j) = base + res;
       base += res;
     }  
   }
   CharacterMatrix<unsigned char> C;
   C.CopyFromMat(M);
   for(MatrixIndexT i = 0; i < M.NumRows() ; i++) {
     for(MatrixIndexT j = 0; j < M.NumCols(); j++) {
        MatrixIndexT x = static_cast<MatrixIndexT>(C(i,j));
        std::cout << x << " ";
     }
     std::cout << "\n";
   }
 }


 } 
int main() {
// CharacterMatrix<float> a(2, 2,1) ;
 //a.SetZero() ;
// a.Set(10);
//a.Resize(10,10,5);
//std::cout<<"next constructor"<<endl;
// a(1,0) = 12 ;
//CharacterMatrix<float> b(2, 2, 4) ; 
// a.Transpose(a) ;
 // b = a ;
 // CharacterMatrix<float> b(a) ; 
 //std::cout<<"row number :"<<a.NumRows()<<" a(0, 1) = "<<a(0,1)<<" a(0, 1) : "<<a(0,1)<< std::endl ;
  kaldi::TestCopyMat<float>();
  std::cout<< "character-matrix-test succeeded.\n";
 return 0 ;
}

