#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <smmintrin.h>//SSE4 intrinscis
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <limits>
#include <math.h>
#include "base/kaldi-error.h"
#include "matrix/kaldi-matrix.h"
#ifndef KALDI_CHARACTER_MATRIX_H_
#define KALDI_CHARACTER_MATRIX_H_
#include "matrix-common.h"
#include "kaldi-blas.h"

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

 private:
  iter  data_;
  MatrixIndexT num_cols_;
  MatrixIndexT num_rows_;
  MatrixIndexT stride_;
  float  min_ ;
  float increment_ ;
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
    Resize(r ,c); 
  }
  // Pegah : CopyFromCharacterMatrix doesn't work! 
  //MatrixTransposeType tM1 = kNoTrans ; 
  CharacterMatrix(const CharacterMatrix<T> & m, MatrixTransposeType trans = kNoTrans) ;//{ CopyFrom(m,MatrixTransposeType tM1 = kNoTrans ); } // copy constructor
  ~CharacterMatrix() {
    free(data_) ; 
    //cout<<"destructor called"<<endl;
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
  void Resize(MatrixIndexT, MatrixIndexT);
  // hhx
  void Transpose();
  template<typename U>
  void AddMatMat(float alpha, 
                 const CharacterMatrix<U> &M1, 
                 MatrixTransposeType tM1, 
                 const CharacterMatrix<T> & M2, 
                 MatrixTransposeType tM2, 
                 const float beta); 
  template<typename Real>
  void  CopyFromMat(const CharacterMatrix<Real> &M, MatrixTransposeType tM = kNoTrans);
  template<typename Real>
  void  CopyFromMatrix(const CharacterMatrix<Real> & M, MatrixTransposeType Trans = kNoTrans) ;  
 // recover real matrix from character matrix
 template<typename Real>
 T R2T(const Real r) ;
 template<typename Real>
 Real T2R(const T t);
  template<typename Real>
  void RecoverMatrix(CharacterMatrix<Real> &M);
};
template<typename T>
CharacterMatrix<T>::CharacterMatrix(const CharacterMatrix<T> & m, MatrixTransposeType trans) { 
  if( trans == kNoTrans) {
    Resize(m.num_rows_, m.num_cols_);
    this->CopyFromMatrix(m);
  } else {
    Resize(m.num_cols_, m.num_rows_);
    this->CopyFromMatrix(m, kTrans) ;
  }
}
template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromMatrix( const CharacterMatrix<Real> & M, MatrixTransposeType Trans) {
 if ( sizeof(T) == sizeof(Real) && (void*) (&M) == (void*)this)
    return ;
 int32 this_stride = stride_, other_stride = M.Stride();
 Real *this_data = data_;
 const Real *other_data = M.begin();
 if ( Trans == kNoTrans) {
  assert( num_rows_ == M.NumRows() && num_cols_ == M.NumCols()) ;
  for (MatrixIndexT i = 0; i < num_rows_; i++) 
    for (MatrixIndexT j = 0; j < num_cols_; j++) 
      this_data[i * this_stride + j] += other_data[i * other_stride + j];
 } else {
   assert( num_rows_ == M.NumCols() && num_cols_ == M.NumRows()) ;
  for (MatrixIndexT i = 0; i < num_rows_; i++) 
    for (MatrixIndexT j = 0; j < num_cols_; j++) 
      this_data[i * this_stride + j] += other_data[j * other_stride + i];
 }
}
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
    //KALDI_ERR << "Failed to do posix memory allot";
    std::cout << "Failed to do posix memory allot";
  }
  //data_ = static_cast<T *> (data);
  // else what?  KALDI_ERROR? [dan]
  num_rows_ = rows;
  num_cols_ = cols;
  stride_  = real_cols;
  this->SetZero();
}

template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromMat(const CharacterMatrix<Real> & M, MatrixTransposeType tM /*= kNoTrans*/) {
  Resize(M.NumRows(),M.NumCols());
  Real min = M.Min(), max = M.Max(); 
       min_ = static_cast<float>(min);

  MatrixIndexT minChar = std::numeric_limits<T>::min(),
        maxChar = std::numeric_limits<T>::max();
  increment_ = static_cast<float>( static_cast<float>(maxChar - minChar)/(max - min));
  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      (*this)(row, col) = R2T<Real>(M(row, col));
    }
  }
}

template<typename T>
template<typename Real>
Real CharacterMatrix<T>::T2R(const T t) {
  MatrixIndexT lower = std::numeric_limits<T>::min();
  Real x = static_cast<Real>(min_ + (t - lower) / increment_);
  return x;
}

template<typename T>
template<typename Real>
T CharacterMatrix<T>::R2T(const Real r) {
  MatrixIndexT lower = std::numeric_limits<T>::min();
  T t = static_cast<T>((r - min_) * increment_ + lower ); 
}

// Recover floating matrix  from char matrix
template<typename T>
template<typename Real>
void CharacterMatrix<T>::RecoverMatrix(CharacterMatrix<Real> &M) {
  M.Resize(num_rows_, num_cols_);
  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      M(row, col) = T2R<Real>((*this)(row, col));
    }
  }
}

template<typename T>
void CharacterMatrix<T>::Transpose() {
  if(num_rows_ != num_cols_) {
    CharacterMatrix<T> tmp(*this, kTrans) ;
    Resize(this->num_cols_, this->num_rows_);
    this->CopyFromMatrix(tmp) ;
  } else {
    MatrixIndexT M = num_rows_;
    for (MatrixIndexT i = 0;i < M;i++)
      for (MatrixIndexT j = 0;j < i;j++) {
        T &a = (*this)(i, j), &b = (*this)(j, i);
        std::swap(a, b);
      }
    }
}


template<typename T>
template<typename U>
void CharacterMatrix<T>::AddMatMat(float alpha, 
                 const CharacterMatrix<U> &M1, 
                 MatrixTransposeType tM1, 
                 const CharacterMatrix<T> & M2, 
                 MatrixTransposeType tM2, 
                 const float beta) {
  KALDI_ASSERT((tM1 == kNoTrans && tM2 == kNoTrans && M1.num_cols_ == M2.num_rows_ && M1.num_rows_ == num_rows_ && M2.num_cols_ == num_cols_)
               || (tM1 == kTrans && tM2 == kNoTrans && M1.num_rows_ == M2.num_rows_ && M1.num_cols_ == num_rows_ && M2.num_cols_ == num_cols_)
               || (tM1 == kNoTrans && tM2 == kTrans && M1.num_cols_ == M2.num_cols_ && M1.num_rows_ == num_rows_ && M2.num_rows_ == num_cols_)
               || (tM1 == kTrans && tM2 == kTrans && M1.num_rows_ == M2.num_cols_ && M1.num_cols_ == num_rows_ && M2.num_rows_ == num_cols_));
  KALDI_ASSERT(&M1 !=  this && &M2 != this);
  float y =  0.0 ;
  if(tM1 == kTrans)
    M1.Transpose();
  if(tM2 == kTrans)
    M2.Transpose();
  for(MatrixIndexT row = 0; row < M1.NumRows(); ++ row) {
    for(MatrixIndexT col = 0; col <M2.NumCols(); ++col) {
      short x = Sse4DotProduct(M1.data_ + row * M1.stride_,
                               M2.data_ + col * M2.stride_, M1.num_cols_);
      float y = alpha * x;
      if(beta==0) {
        (*this)(row, col) = R2T<float>(y);
      } else {
        float this_x = static_cast<float>(T2R((*this)(row, col)));
         y +=  beta * this_x;
        (*this)(row, col) = R2T<float>(y);
      }
    }
  }
}

template <typename T>
CharacterMatrix<T>& CharacterMatrix<T>::operator = (const CharacterMatrix& rhs) {
  if (CharacterMatrix<T>::NumRows() != rhs.NumRows() || CharacterMatrix<T>::NumCols() != rhs.NumCols()) {
    Resize(rhs.NumRows(), rhs.NumCols());
  }
  std::cout<<" we are here in operator= "<<std::endl ;
  CharacterMatrix<T>::CopyFromMat(rhs);
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

} //  mamespace kaldi

#endif  // KALDI_CHARACTER_MATRIX_H_

