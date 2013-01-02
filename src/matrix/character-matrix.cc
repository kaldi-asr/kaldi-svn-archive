#include "character-matrix.h"
#include <iostream>
int main() {
  CharacterMatrix<float> a(2, 2,1);
  a.SetZero();
  a.Set(10);
  a.Resize(4,3,5);
  std::cout << "The row number is : " << a.NumRows() << " a(0, 0) = " << a(0,0)<< std::endl ;
  CharacterMatrix<float> b(4,3,0.2);
  CharacterMatrix<float> c(3,4,0);
  CharacterMatrix<int> d(4,3,5) ;
  //std::cout<<" d after change :"<<CharacterMatrix<float>::CopyFromCharacterMatrix2(b)<<std::endl;
  d = CharacterMatrix<float>::CopyFromCharacterMatrix2(b) ;
  std::cout<<" d(1,1) is : "<<d(1,1) << std::endl ;
  c.Transpose(b);
  float fVal = 60 ;
  char cVal[32] ;
  sprintf(cVal, "%f", fVal) ;
  std::cout<<"cVal :"<< cVal<<std::endl ;
  //std::cout << "row number : " << c.NumRows() << " c(2, 3) = " << c(2,3)<< std::endl ;
  return 0;
}
