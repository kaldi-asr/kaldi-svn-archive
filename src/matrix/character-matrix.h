#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <smmintrin.h>//SSE4 intrinscis
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <limits>
#include <math.h>
#include "matrix/kaldi-matrix.h"

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
namespace  kaldi {

inline int DotProduct(unsigned char *x, signed char *y, MatrixIndexT length) {
 int i;
 int sum=0;
    for(i=0; i< length; i++) {
        sum += x[i] * y[i];
    }
    return sum; 
}
inline  int Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length) {
 /*   int i;
    __m128i c, lo, hi;
    __m128i *e, *f;
    __m128i sum = _mm_setzero_si128();
    int result;
    
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
    
    return result;   */


    int i;
    __m128i c1, lo1, hi1, c2, lo2, hi2, c3, lo3, hi3, c4, lo4, hi4;
    __m128i *e1, *f1, *e2, *f2, *e3, *f3, *e4, *f4;
    __m128i sum1 = _mm_setzero_si128();
    __m128i sum2 = _mm_setzero_si128();
    __m128i sum3 = _mm_setzero_si128();
    __m128i sum4 = _mm_setzero_si128();

    __m128i sum = _mm_setzero_si128();
    
     int result;
    
    for (i=0; i+63 < length; i+=64) {
        e1 = (__m128i*)(x+i);
        f1 = (__m128i*)(y+i);
        e2 = (__m128i*)(x+i+16);
        f2 = (__m128i*)(y+i+16);
        e3 = (__m128i*)(x+i+32);
        f3 = (__m128i*)(y+i+32);
        e4 = (__m128i*)(x+i+48);
        f4 = (__m128i*)(y+i+48);
        
        c1 = _mm_maddubs_epi16(e1[0], f1[0]); // performs dot-product in 2X2 blocks
        c2 = _mm_maddubs_epi16(e2[0], f2[0]); // performs dot-product in 2X2 blocks
        c3 = _mm_maddubs_epi16(e3[0], f3[0]); // performs dot-product in 2X2 blocks
        c4 = _mm_maddubs_epi16(e4[0], f4[0]); // performs dot-product in 2X2 blocks
        
        // unpack the 4 lowest 16-bit integers into 32 bits.
        lo1 = _mm_cvtepi16_epi32(c1);
        lo2 = _mm_cvtepi16_epi32(c2);
        lo3 = _mm_cvtepi16_epi32(c3);
        lo4 = _mm_cvtepi16_epi32(c4);

        // unpack the 4 highest 16-bit integers into 32 bits.
        hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));

        sum1 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum1);  // pass the result to sum
        sum2 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum2);  // pass the result to sum
        sum3 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum3);  // pass the result to sum
        sum4 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum4);  // pass the result to sum

    }
    for (; i<length; i+=16) {
        e1 = (__m128i*)(x+i);
        f1 = (__m128i*)(y+i);
        c1 = _mm_maddubs_epi16(e1[0], f1[0]); // performs dot-product in 2X2 blocks
        
        // unpack the 4 lowest 16-bit integers into 32 bits.
        lo1 = _mm_cvtepi16_epi32(c1);
        // unpack the 4 highest 16-bit integers into 32 bits.
        hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum1);  // pass the result to sum
    }
    
    
    
    sum = _mm_add_epi32(_mm_add_epi32(sum1, sum2), _mm_add_epi32(sum3, sum4));
    sum = _mm_hadd_epi32(sum,sum); // perform horizontal addition to sum up the partial dot-products
    sum = _mm_hadd_epi32(sum,sum); // perform horizontal addition to sum up the partial dot-products
    
    result = _mm_cvtsi128_si32(sum); // extract dot-product result by moving the least significant 32 bits of the 128-bit "sum" to a 32-bit integer result
    
    return result;  
 
}

inline int Sse4SumArray(unsigned char *x, MatrixIndexT length) {
  int i;
  __m128i lo, hi;
  __m128i *e;
  __m128i c = _mm_setzero_si128();
  //__m128i tmpsum = _mm_setzero_si128();
  __m128i sum = _mm_setzero_si128();

  // const __m128i vk0 = _mm_set1_epi8(0);       // constant vector of all 0s for use with _mm_unpacklo_epi8/_mm_unpackhi_epi8
  int result;

  for (i=0; i<length; i+=16) {
    e = (__m128i*)(x+i);

    // unpack the 8 lowest 8-bit integers into 16 bits.
    // lo = _mm_unpacklo_epi8(e[0],vk0);
    lo = _mm_cvtepi8_epi16(e[0]);
    // unpack the 8 highest 8-bit integers into 16 bits.
    //   hi = _mm_unpackhi_epi8(e[0],vk0);
    hi = _mm_cvtepi8_epi16(_mm_shuffle_epi32(e[0], 0x4e));

    c = _mm_hadd_epi16(lo, hi);

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

inline int Sse4SumArray(signed char *x, MatrixIndexT length) {
  int i;
  __m128i lo, hi;
  __m128i *e;
  __m128i c = _mm_setzero_si128();
  //__m128i tmpsum = _mm_setzero_si128();
  __m128i sum = _mm_setzero_si128();

  // const __m128i vk0 = _mm_set1_epi8(0);       // constant vector of all 0s for use with _mm_unpacklo_epi8/_mm_unpackhi_epi8
  int result;

  for (i=0; i<length; i+=16) {
    e = (__m128i*)(x+i);

    // unpack the 8 lowest 8-bit integers into 16 bits.
    // lo = _mm_unpacklo_epi8(e[0],vk0);
    lo = _mm_cvtepi8_epi16(e[0]);
    // unpack the 8 highest 8-bit integers into 16 bits.
    //   hi = _mm_unpackhi_epi8(e[0],vk0);
    hi = _mm_cvtepi8_epi16(_mm_shuffle_epi32(e[0], 0x4e));

    c = _mm_hadd_epi16(lo, hi);

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

template<typename T>
class CharacterMatrix{

 // from Dan: these types are not needed.
 typedef T* iter;
 typedef const T* const_iter;
  
 template <typename Real>friend class MatrixBase;

  // from Dan: Google style guide demands that the public section be first.
 private:
  iter  data_;
  MatrixIndexT num_cols_;
  MatrixIndexT num_rows_;
  MatrixIndexT stride_;
  // from Dan: consider changing this to alpha_ and beta_ representation, which
  // will be simpler in AddMatMat.
  float min_ ;
  float incremental_ ;
  // From Dan: Google style guide does not allow such un-informative names.
  // Consider CharToReal and RealToChar.  And they should be inline functions for speed.
   template<typename Real>
  inline Real CharToReal(const T t);
  
  template<typename Real>
  inline T RealToChar(const Real r);
 public:
  //constructors & destructor:
  CharacterMatrix() {
    data_ = 0;
    num_rows_ = 0;
    num_cols_ = 0;
    stride_  = 0;  
  } 

  
  // make it explicit to make statement like "vec<int> a = 10;" illegal.
  // no need for "explicit" if it takes >1 argument. [dan]
  CharacterMatrix(MatrixIndexT r, MatrixIndexT c, const T& value = T()) { 
    Resize(r, c);
  }
  
  // Pegah : CopyFromCharacterMatrix doesn't work!
  // From Dan: what is "kNoTrans" still doing here?
  CharacterMatrix(const CharacterMatrix& m) { } // copy constructor
  ~CharacterMatrix() { 
    //cout<<"destructor called"<<endl;
    free(data_);
  } 
  //operator overloading functions:
    
  CharacterMatrix& operator = (const CharacterMatrix&); // assignment operator

  // From Dan: It's OK to define this, but you should not call it from any functions you want to be fast.
  // Also, I would probably rather have it return real, as we want this class to be such that "externally"
  // it acts like it stores real.
  T&  operator() (MatrixIndexT r, MatrixIndexT c) {
   //std::cout<<" r : "<<r<<" c : "<<c<<" num rows : "<<num_rows_<<" um cols : "<<num_cols_<<std::endl ;
   //assert(r < num_rows_ && c < num_cols_) ;
   return *(data_ + r * stride_ + c);
  }
  const  T&  operator() (MatrixIndexT r, MatrixIndexT c) const {
    return *(data_ + r * stride_ + c);
  }
  
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_cols_; }
  inline MatrixIndexT Stride() const { return stride_; }
  inline iter begin() const { return data_; }
  // [dan]: delete clear() and empty().  We can use Resize(0, 0).
  //void
  //bool empty() const { return num_rows_ == 0 || num_cols_ == 0; }
  void SetZero();  
  // from Dan: you can remove the last argument to Resize; try to make the
  // interface like that of Matrix, where Resize takes a typedef (look at it.)
  void Resize(MatrixIndexT, MatrixIndexT);

  // From Dan: the following function probably won't be needed, but one day it
  // might be useful so it's OK to define it.  If you initialize with kTrans from
  // a real-valued matrix, the initialization code should do the transposing itself,
  // so it can be efficient.
  void Transpose();
  template<typename Real>
  void  CopyFromMat(const MatrixBase<Real> &M);

  // recover real matrix from character matrix
  // From Dan: Google style guide does not allow non-const references, you should use a pointer.
  // But this should probably be called CopyToMat instead of RecoverMatrix.
  template<typename Real>
  void CopyToMat(Matrix<Real> *M);
  // test, to be removed
  float T2R(const T &t);
};
 
template<typename T>
void CharacterMatrix<T>::SetZero() {
  memset(data_, 0, sizeof(T)*num_rows_*stride_);
}

template<typename T>
void CharacterMatrix<T>::Resize(MatrixIndexT rows, MatrixIndexT cols)
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
  int pos_ret = posix_memalign(static_cast<void**>(&data), 16, size);
  if(pos_ret != 0) {
    KALDI_ERR << "Failed to do posix memory allot";
  }
  data_ = static_cast<T *> (data);
  // else what?  KALDI_ERROR? [dan]
  num_rows_ = rows;
  num_cols_ = cols;
  stride_  = real_cols;
  this->SetZero();
}

template<typename T>
template<typename Real>
inline Real CharacterMatrix<T>::CharToReal(const T t) {
  int32 lower = std::numeric_limits<T>::min();
  Real x = static_cast<Real>(min_ + (t - lower) / incremental_);
  return x;
}

template<typename T>
template<typename Real>
T CharacterMatrix<T>::RealToChar(const Real r) {
  int32 lower = std::numeric_limits<T>::min();
  T t = static_cast<T>((r - min_) * incremental_ + lower);
  //if((r - min_) * incremental_ + lower - t - 0.5)
// std::cout << " t before = "<<static_cast<int>(t)<<std::endl ;
// std::cout << " (r - min_) * incremental_ + lower = " << (r - min_) * incremental_ + lower << std::endl ; 
 if(std::abs((r - min_) * incremental_ + lower - static_cast<Real>(t)) > 0.5)  {
  if( (r - min_) * incremental_ + lower < static_cast<Real>(0)) {
    t = t - static_cast<T>(1) ;
  } else {
    t = t + static_cast<T>(1) ;
  }
 }
 //std::cout << " t after = "<<static_cast<int>(t)<<std::endl ;
  return t; 
}
// test function, to be removed
template<typename T>
float CharacterMatrix<T>::T2R(const T &t) {
  int32 lower = std::numeric_limits<T>::min();
  float x = static_cast<float>(min_ + (t - lower) / incremental_);
  return x;
}

template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromMat(const MatrixBase<Real> & M) {
  Resize(M.NumRows(),M.NumCols());
  Real min = M.Min(), max = M.Max(); 
       min_ = static_cast<float>(min);

  int32 minChar = std::numeric_limits<T>::min(),
        maxChar = std::numeric_limits<T>::max();
  incremental_ = static_cast<float>( static_cast<float>(maxChar - minChar)/(max - min));

  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      (*this)(row, col) = RealToChar<Real>(M(row, col));
    }
  }
}

// Recover floating matrix  from char matrix
template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyToMat(Matrix<Real> *M) {
  M->Resize(num_rows_, num_cols_);
  for (MatrixIndexT row = 0; row < M->NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M->NumCols(); col++) {
      (*M)(row, col) = CharToReal<Real>((*this)(row, col));
    }
  }
}

template <typename T>
CharacterMatrix<T>& CharacterMatrix<T>::operator = (const CharacterMatrix& rhs) {
  if (CharacterMatrix<T>::NumRows() != rhs.NumRows() || CharacterMatrix<T>::NumCols() != rhs.NumCols()) {
    Resize(rhs.NumRows(), rhs.NumCols());
  }
  std::cout<<" we are here in operator= "<<std::endl ;
  CharacterMatrix<T>::CopyFromCharacterMatrix(rhs,"kNoTrans");
  return *this;
}


} //  namespace kaldi

#endif  // KALDI_CHARACTER_MATRIX_H_

