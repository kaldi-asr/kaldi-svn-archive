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

inline int Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length) {
  int i;
  __m128i c, d, lo, hi;
  __m128i *e, *f;
  __m128i sum = _mm_setzero_si128();
  int result;
  
  for (i = 0; i < length; i += 16) {
    e = (__m128i*)(x+i);
    f = (__m128i*)(y+i);
    c = _mm_maddubs_epi16(e[0], f[0]); // performs dot-product in 2X2 blocks

    // unpack the 4 lowest 16-bit integers into 32 bits.
    lo = _mm_cvtepi16_epi32(c);


    // unpack the 4 highest 16-bit integers into 32 bits.
    hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
    d = _mm_add_epi32(lo, sum);    
    sum = _mm_add_epi32(d, hi);  // pass the result to sum
  }
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
  Real T2R(const T t);
  template<typename Real>
  T R2T(const Real r);
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
    Resize(r, c ,value); 
  }
  
  // Pegah : CopyFromCharacterMatrix doesn't work!
  // From Dan: what is "kNoTrans" still doing here?
  CharacterMatrix(const CharacterMatrix& m) { CopyFromCharacterMatrix(m, "kNoTrans"); } // copy constructor
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
   assert(r < num_rows_ && c < num_cols_) ;
   return *(data_ + r * stride_ + c);
  }
  const  T&  operator() (MatrixIndexT r, MatrixIndexT c) const {
    return *(data_ + r * stride_ + c);
  }
  // From Dan: delete this function:
  inline iter begin() const { return data_ ; } 
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_cols_; }
  inline MatrixIndexT Stride() const { return stride_; }
  // From Dan: the next 2 functions are probably not going to be needed.
  T Min() const ;
  T Max() const ;
  // [dan]: delete clear() and empty().  We can use Resize(0, 0).
  //void
  //bool empty() const { return num_rows_ == 0 || num_cols_ == 0; }
  void SetZero();
  // from Dan: the next function will probably not be needed.
  void Set(T value);
    
  // from Dan: you can remove the last argument to Resize; try to make the
  // interface like that of Matrix, where Resize takes a typedef (look at it.)
  void Resize(MatrixIndexT, MatrixIndexT, const T&);

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
  void RecoverMatrix(Matrix<Real> &M);
 
};
 
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
  int pos_ret = posix_memalign(static_cast<void**>(&data), 16, size);
  if(pos_ret != 0) {
    KALDI_ERR << "Failed to do posix memory allot";
  }
  data_ = static_cast<T *> (data);
  // else what?  KALDI_ERROR? [dan]
  num_rows_ = rows;
  num_cols_ = cols;
  stride_  = real_cols;
  this->Set(value);
}

template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromMat(const MatrixBase<Real> & M) {
  Resize(M.NumRows(),M.NumCols(),0);
  Real min = M.Min(), max = M.Max(); 
       min_ = static_cast<float>(min);

  int32 minChar = std::numeric_limits<T>::min(),
        maxChar = std::numeric_limits<T>::max();
  incremental_ = static_cast<float>( static_cast<float>(maxChar - minChar)/(max - min));

  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      (*this)(row, col) = R2T<Real>(M(row, col));
    }
  }
}

template<typename T>
template<typename Real>
Real CharacterMatrix<T>::T2R(const T t) {
  int32 lower = std::numeric_limits<T>::min();
  Real x = static_cast<Real>(min_ + (t - lower) / incremental_);
  return x;
}

template<typename T>
template<typename Real>
T CharacterMatrix<T>::R2T(const Real r) {
  int32 lower = std::numeric_limits<T>::min();
  T t = static_cast<T>((r - min_) * incremental_ + lower);
  return t; 
}

// Recover floating matrix  from char matrix
template<typename T>
template<typename Real>
void CharacterMatrix<T>::RecoverMatrix(Matrix<Real> &M) {
  M.Resize(num_rows_, num_cols_);
  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      M(row, col) = T2R<Real>((*this)(row, col));
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


} //  namespace kaldi

#endif  // KALDI_CHARACTER_MATRIX_H_

