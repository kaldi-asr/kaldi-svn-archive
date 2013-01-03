#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <smmintrin.h>//SSE4 intrinscis
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <limits>
#include <math.h>
//#include "matrix/kaldi-matrix.h"
#ifndef KALDI_CHARACTER_MATRIX_H_
#define KALDI_CHARACTER_MATRIX_H_

//a trial of Matrix class with SSE multiplication. By Xiao-hui Zhang 2012/12
// The code was debugged by Pegah and Ehsan 12.28.12
// The CopyFromMat function create these values appropriately.
// You can compare with the CompressedMatrix code which does something similar
// [but slightly more complicate.]
// 
// You can get the min and max values of your integer range (e.g. 0 and 255, or -128 and +127),
//  from std::numeric_limits, to make your code generic.

// You'll probably have just one matrix-matrix multiplication, which would be
// a class member of Matrix<Real>: something like:
// void AddMat(Real alpha, const CharacterMatrix<char> &A, MatrixTransposeType tA,
//             const CharacterMatrix<unsigned char> &B, MatrixTransposeType tB, Real beta);
// and maybe one called:
// void AddMat(Real alpha, const CharacterMatrix<unsigned char> &A, MatrixTransposeType tA,
//             const CharacterMatrix<char> &B, MatrixTransposeType tB, Real beta);
// and you don't have to support all values of tA and tB-- you can just support whichever
// one would be most efficient to implement, and crash if the user gives other values.
// you could declare that function in kaldi-matrix.h but define it in character-matrix.cc
// You may have to use "friend declarations" here, to make this work.
namespace kaldi {
template<typename T>
class CharacterMatrix{

 typedef T* iter;
 typedef const T* const_iter;
 typedef int32_t MatrixIndexT;
 typedef std::string MatrixTransposeType;
 private:
  iter  data_;
  MatrixIndexT num_cols_;
  MatrixIndexT num_rows_;
  MatrixIndexT stride_;
  float  min_ ;
  float incremental_ ;
  // from Dan: if you need this function it should be called Sse4DotProduct.
  // but it probably doesn't belong here, e.g. could be a static inline function
  // declared and defined in character-matrix.cc.
  short int  Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length);

 public:
  //constructors & destructor:
  CharacterMatrix() {
    data_ = 0;
    num_rows_ = 0;
    num_cols_ = 0;
    stride_  = 0;  
  } 
  // note from Dan: this create() function is only
  // called once so put the code here unless you have other plans-- also, it should
  // have been called Init() if you had had it.
  
  // make it explicit to make statement like "vec<int> a = 10;" illegal.
  // no need for "explicit" if it takes >1 argument. [dan]
  CharacterMatrix(MatrixIndexT r, MatrixIndexT c, const T& value = T()) { 
    //cout<<"constructor called"<<endl;
    Resize(r ,c ,value); 
  }
  // Pegah : CopyFromCharacterMatrix doesn't work!  
  CharacterMatrix(const CharacterMatrix& m) { CopyFromCharacterMatrix(m, "kNoTrans"); } // copy constructor
  ~CharacterMatrix() { 
    //cout<<"destructor called"<<endl;
    free(data_);
  } 
  //operator overloading functions:
    
  CharacterMatrix& operator = (const CharacterMatrix&); // assignment operator

  T&  operator() (MatrixIndexT r, MatrixIndexT c) {
   //std::cout<<" r : "<<r<<" c : "<<c<<" num rows : "<<num_rows_<<" um cols : "<<num_cols_<<std::endl ;
   assert(r < num_rows_ && c < num_cols_) ;
   return *(data_ + r * stride_ + c);
  }
  const  T&  operator() (MatrixIndexT r, MatrixIndexT c) const {
    return *(data_ + r * stride_ + c);
  }
  inline iter begin() const { return data_ ; } 
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_cols_; }
  inline MatrixIndexT Stride() const { return stride_; }
  T Min() const ;
  T Max() const ;
  // [dan]: delete clear() and empty().  We can use Resize(0, 0).
  //void
  //bool empty() const { return num_rows_ == 0 || num_cols_ == 0; }
  void SetZero();
  void Set(T value);
    
  void Resize(MatrixIndexT, MatrixIndexT, const T&);
  template<typename Real>
  void Transpose(const CharacterMatrix<Real> & M);
  void AddMatMat(T alpha, const CharacterMatrix<unsigned char> & M1, MatrixTransposeType tM1, const CharacterMatrix<signed char> & M2, MatrixTransposeType tM2, T beta); 
  template<typename Real>
  void CopyFromCharacterMatrix(const CharacterMatrix<Real> & M, MatrixTransposeType  tM);
// modified by hhx
 template<typename Real>
 void CopyFromMat(const CharacterMatrix<Real> &M, MatrixTransposeType  tM) ;
// Recover the float matrix
 template<typename Real>
 void RecoverMatrix(CharacterMatrix<Real> &M) ; 
} ;

template<typename T>
void CharacterMatrix<T>::SetZero() {
  if (num_cols_ == stride_){
    memset(data_, 0, sizeof(T)*num_rows_*num_cols_);
  } else {
    for (MatrixIndexT row = 0; row < num_rows_; row++) {
      memset(data_ + row*stride_, 0, sizeof(T)*num_cols_);
    }
  }
}
template<class T>
T CharacterMatrix<T>::Max() const {
  assert(num_rows_ > 0 && num_cols_ > 0);
  T ans= *data_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      if (data_[c + stride_*r] > ans)
        ans = data_[c + stride_*r];
  return ans;
}

template<class T>
T CharacterMatrix<T>::Min() const {
  assert(num_rows_ > 0 && num_cols_ > 0);
  T ans= *data_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      if (data_[c + stride_*r] < ans)
        ans = data_[c + stride_*r];
  return ans;
}

template<typename T>
void CharacterMatrix<T>::Set(T value) {
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    for (MatrixIndexT col = 0; col < num_cols_; col++) {
      (*this)(row, col) = value;
    }
  }
}

template<typename T>
void CharacterMatrix<T>::Resize(MatrixIndexT rows, MatrixIndexT cols, const T& value)
{
  MatrixIndexT skip;
  MatrixIndexT real_cols;
  size_t size;
  void*   data;       // aligned memory block
  
  // compute the size of skip and real cols
  skip = ((16 / sizeof(T)) - cols % (16 / sizeof(T))) % (16 / sizeof(T));
  real_cols = cols + skip; 
  // Pegah : sizeof(Real) changed to sizeof(T); I think it is not reuired to have sizeof(T), since it will be multiplied in posix-memolign
  size = static_cast<size_t>(rows) * static_cast<size_t>(real_cols) * sizeof(T);
    
  // allocate the memory and set the right dimensions and parameters
  // WARNING from Dan: you should not put code that you need to run, inside an
  // assert.  If you compile with -ndebug (no-debug), it will not get executed.
  assert(posix_memalign(static_cast<void**>(&data), 16, size) == 0 ); 
  data_ = static_cast<T *> (data);
  // else what?  KALDI_ERROR? [dan]
  num_rows_ = rows;
  num_cols_ = cols;
  stride_  = real_cols;
  this->Set(value);
}

template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromCharacterMatrix(const CharacterMatrix<Real> & M, MatrixTransposeType  tM) {
    if(tM.compare("kNoTrans")!= 0 ){ 
	M.Transpose(M);
	std::cout<<" we are in kTrans mode"<<std::endl ;
    } else {
    MatrixIndexT this_stride = stride_, other_stride = M.Stride() ;
    T *this_data = data_ ;
    const Real *other_data = M.begin();
    for(MatrixIndexT row = 0; row < num_rows_; row++) {
     for (MatrixIndexT col = 0; col < num_cols_; col++) {
  //     sprintf(tmp1(row, col), "%f",M(row, col));
	this_data[row * this_stride+ col] = other_data[row * other_stride + col] ;
	//(*this)(row, col) =M(row, col);
	
    }
  }
  }
}

template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromMat(const CharacterMatrix<Real> & M, MatrixTransposeType  tM) {
  Resize(M.NumRows(), M.NumCols(),0) ;
  Real min = M.Min();
  Real  max = M.Max();
  Real  min_ = static_cast<float>(min);
  MatrixIndexT minChar = std::numeric_limits<T>::min(),maxChar = std::numeric_limits<T>::max();
  incremental_ = static_cast<float>( static_cast<float>(maxChar - minChar)/(max - min));
  if ( tM.compare("kNoTrans") == 0 ) {
//  std::cout<< "we are in CopyFromMat"<<" M rows : "<<M.NumRows()<<" C rows :"<<num_rows_<<std::endl ;
  assert(num_rows_ == M.NumRows() && num_cols_ == M.NumCols()) ;
	
  MatrixIndexT this_stride = stride_, other_stride = M.Stride() ;
  T *this_data = data_ ;
  const Real *other_data = M.begin() ;
  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
    // (*this)(row, col) = static_cast<T>( (M( row, col) - min_) * incremental_  + minChar );
    this_data[row * this_stride + col] =static_cast<T>((other_data[row * other_stride + col]-min_) * incremental_  + minChar ) ;
    }
  }
}
}
// Recover floating matrix  from char matrix
template<typename T>
template<typename Real>
void CharacterMatrix<T>::RecoverMatrix(CharacterMatrix<Real> &M) {
  M.Resize(num_rows_, num_cols_);
  MatrixIndexT minChar = std::numeric_limits<T>::min();
  MatrixIndexT maxChar = std::numeric_limits<T>::max(), range = maxChar - minChar;
  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      M(row, col) = static_cast<Real>(min_ + ((*this)(row,col) - minChar) / incremental_);
    }
  }
} 
template<typename T>
template<typename Real>
void CharacterMatrix<T>::Transpose(const CharacterMatrix<Real> & M){
  (*this).Resize(M.NumCols(), M.NumRows(), 0);
  for (MatrixIndexT row = 0; row < M.NumCols(); row++) {
    for (MatrixIndexT col = 0; col < M.NumRows(); col++) {
      (*this)(row, col) = static_cast<T> (M(col, row));
    }
  }
}


template<typename T>
void CharacterMatrix<T>::AddMatMat(T alpha, const CharacterMatrix<unsigned char> & M1, MatrixTransposeType tM1, const CharacterMatrix<signed char> & M2, MatrixTransposeType tM2, T beta){
  if( tM1.compare("kNoTrans") !=0) M1.Transpose(M1);
  if( tM2.compare("kNoTrans") != 0) M2.Transpose(M2);
  assert(M1.NumCols() == M2.NumRows());
  (*this).Resize(M1.NumRows(), M2.NumCols(), 0);
  CharacterMatrix<signed char> M2T;
  M2T.Transpose(M2);
  short tmp;
  for (MatrixIndexT row = 0; row < M1.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M2.NumCols(); col++) {
      tmp = Sse4DotProduct((unsigned char *)(M1.begin() + row * M1.Stride(),(signed char *)(M2T.begin() + col * M2T.Stride(), M1.NumCols())));
      (*this)(row, col) = alpha * static_cast<T> (tmp)+beta * (*this)(row, col);
    }
  }
}

template <typename T>
CharacterMatrix<T>& CharacterMatrix<T>::operator = (const CharacterMatrix& rhs) {
  if (CharacterMatrix<T>::NumRows() != rhs.NumRows() || CharacterMatrix<T>::NumCols() != rhs.NumCols()) {
    Resize(rhs.NumRows(), rhs.NumCols(), 0);
  }
  std::cout<<" we are here in operator= "<<std::endl ;
  CharacterMatrix<T>::CopyFromCharacterMatrix(rhs,"kNoTrans");
  return *this;
}

template<typename T>
short int CharacterMatrix<T>::Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length)
{
  int i;
  __m128i a, b, c, lo, hi;
  __m128i *e, *f;
  __m128i sum = _mm_setzero_si128();
  short result;
  
  for (i=0; i<length; i+=16) {
    e = (__m128i*)(x+i);
    f = (__m128i*)(y+i);
    c = _mm_maddubs_epi16(e[0], f[0]); // performs dot-product in 2X2 blocks
    
    // unpack the 4 lowest 16-bit integers into 32 bits.
    lo = _mm_cvtepi16_epi32(c);
    // unpack the 4 highest 16-bit integers into 32 bits.
    hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);  // pass the result to sum
  }
  sum = _mm_hadd_epi32(sum,sum); // perform horizontal addition to sum up the partial dot-products
  sum = _mm_hadd_epi32(sum,sum); // perform horizontal addition to sum up the partial dot-products
  
  result = _mm_cvtsi128_si32(sum); // extract dot-product result by moving the least significant 32 bits of the 128-bit "sum" to a 32-bit integer result
  
  return result;
}
} // namespace kaldi
#endif  // KALDI_CHARACTER_MATRIX_H_
