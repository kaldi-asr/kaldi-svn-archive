#include "character-matrix.h"
#include <iostream>
int main() {
 CharacterMatrix<float> a(2, 2,1) ;
 a.SetZero() ;
 a.Set(10);
 a.Resize(10,10,5);
// a(1,0) = 12 ;
//CharacterMatrix<float> b(2, 2, 4) ; 
// a.Transpose(a) ;
 // b = a ;
 // CharacterMatrix<float> b(a) ; 
 std::cout<<"row number :"<<a.NumRows()<<" a(0, 1) = "<<a(0,1)<<" a(0, 1) : "<<a(0,1)<< std::endl ;
 return 0 ;
}

