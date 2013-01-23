#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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
template<typename Real> class CharacterMatrixBase;
template<typename Real> class CharacterSubMatrix;
template<typename Real> class CharacterMatrix;

int DotProduct(unsigned char *x, signed char *y, MatrixIndexT length);

int DotProduct2(unsigned char *x, signed char *y, MatrixIndexT length);

int Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length);

void Sse4DotProduct4fold1X4(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length);

void Sse4DotProduct8fold1X4V3(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length);

void Sse4DotProduct8fold1X4V2(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length);

void Sse4DotProduct8fold1X4(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length);

int Sse4SumArray(unsigned char *x, MatrixIndexT length);

template<typename T>
class CharacterMatrixBase {
 public:
  friend class MatrixBase<T>;
  friend class CharacterSubMatrix<T>;
  friend class CharacterMatrix<T>;
 
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_cols_; }
  inline MatrixIndexT Stride() const { return stride_; }
  inline  T* Data() const { return data_; }
  inline float Min() const { return min_; }
  inline float Increment() const { return increment_; }
  void SetZero();  
  void SetValue(const T t); 
 
 //operator overloading functions:
  
  inline T&  operator() (MatrixIndexT r, MatrixIndexT c) {
   //assert(r < num_rows_ && c < num_cols_) ;
   return *(data_ + r * stride_ + c);
  }
  inline const  T  operator() (MatrixIndexT r, MatrixIndexT c) const {
    return *(data_ + r * stride_ + c);
  }
 /* 
  template<typename Real>
  void  CopyFromMat(const MatrixBase<Real> &M);
  
  template<typename OtherReal>
  void CopyFromMatrix(const CharacterMatrixBase<OtherReal> &M);
  */
  // recover real matrix from character matrix
  // From Dan: Google style guide does not allow non-const references, you should use a pointer.
  // But this should probably be called CopyToMat instead of RecoverMatrix.

  template<typename Real>
  void CopyToMat(Matrix<Real> *M);
 
 // it tells 
  void BlockResize(const int32 blk_num_rows, const int32 blk_num_cols) {
    if (blk_num_rows <= 0 || blk_num_cols <= 0) {
      KALDI_ERR << "blocking size error";
    }
    blk_num_rows_ = blk_num_rows;
    blk_num_cols_ = blk_num_cols;
  }
  // test, to be removed
  float T2R(const T &t);
  template<typename Real>
  inline Real CharToReal(const T t);
  
  template<typename Real>
  inline T RealToChar(const Real r);
  
 void CheckMatrix(); 
 
 CharacterSubMatrix<T> Range(const MatrixIndexT ro, const MatrixIndexT r, 
                             const MatrixIndexT col,const MatrixIndexT c);
 protected:
  CharacterMatrixBase() { }
  ~CharacterMatrixBase() { }
  inline T* Data_workaround() const {
    return data_;
  }

  // for matrix blocking
  int32 blk_num_rows_;  // rows of the blocking matrix 
  int32 blk_num_cols_;  // columns of the blocking matrix
  int32 row_blks_;      //  ( num_rows_ + padding row number ) / blk_num_rows_
  int32 col_blks_;      //  (num_cols_ + padding column number) / blk_num_cols_ 
  // remember we always pad column, then pad row, if any.
  // I think this is concerned with that we favor row-major matrix base
  // matrix-product, and we don't want to separately deal with
  // corner case [hhx]
  

  T*  data_;
  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;
  MatrixIndexT stride_;
  // from Dan: consider changing this to alpha_ and beta_ representation, which
  // will be simpler in AddMatMat.
  float min_ ;
  float increment_ ;
 

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CharacterMatrixBase);  

};

template<typename T>
class CharacterMatrix : public CharacterMatrixBase<T>{

  
 template <typename Real>friend class MatrixBase;

 public:
  //constructors & destructor:
  CharacterMatrix() {CharacterMatrixBase<T>::blk_num_cols_ = 0;
                    CharacterMatrixBase<T>::row_blks_ = 0;
                    CharacterMatrixBase<T>::col_blks_ = 0;
                    CharacterMatrixBase<T>::data_ = NULL;
                    CharacterMatrixBase<T>::num_rows_ = 0;
                    CharacterMatrixBase<T>::num_cols_ = 0;
                    CharacterMatrixBase<T>::stride_ = 0;
                    CharacterMatrixBase<T>::blk_num_rows_ = 0;} 

  
  // make it explicit to make statement like "vec<int> a = 10;" illegal.
  // no need for "explicit" if it takes >1 argument. [dan]
  CharacterMatrix(MatrixIndexT r, MatrixIndexT c, const T& value = T()) { 
    Resize(r, c);
  }
 // template<typename Real>
 // CharacterMatrix(const CharacterMatrix<Real> &M) { CopyFromMatrix(M);} 
  // Pegah : CopyFromCharacterMatrix doesn't work!
  // From Dan: what is "kNoTrans" still doing here?
  explicit CharacterMatrix(const CharacterMatrixBase<T>& M,
                           MatrixTransposeType trans = kNoTrans); // copy constructor

  
  ~CharacterMatrix() { 
    //cout<<"destructor called"<<endl;
    free(CharacterMatrixBase<T>::data_);
  } 
  
  // operator overloading

  CharacterMatrix<T> &operator = (const CharacterMatrixBase<T> &other) {
    if ( CharacterMatrixBase<T>::NumRows() != other.NumRows() ||
         CharacterMatrixBase<T>::NumCols() != other.NumCols()) {
      Resize(other.NumRows(), other.NumCols());
    }
    CharacterMatrixBase<T>::CopyFromMatrix(other);
    return *this;
  }
  
  // from Dan: you can remove the last argument to Resize; try to make the
  // interface like that of Matrix, where Resize takes a typedef (look at it.)
  void Resize(MatrixIndexT, MatrixIndexT);
  //void CheckMatrix();
  // From Dan: the following function probably won't be needed, but one day it
  // might be useful so it's OK to define it.  If you initialize with kTrans from
  // a real-valued matrix, the initialization code should do the transposing itself,
  // so it can be efficient.
  void Transpose();
 
  template<typename Real>
  void  CopyFromMat(const MatrixBase<Real> &M);
  
  template<typename OtherReal>
  void CopyFromMatrix(const CharacterMatrix<OtherReal> &M);
  
 protected:

 private:

};
 
template<typename T>
void CharacterMatrixBase<T>::SetZero() {
  if(blk_num_rows_ == 0) {
    memset(data_, 0, sizeof(T)*num_rows_*stride_);
  } else {
    int32 mem_rows = blk_num_rows_ * row_blks_;
    memset(data_, 0, sizeof(T) * mem_rows * stride_);
  }
}
template<typename T>
void CharacterMatrixBase<T>::SetValue(const T t) {
  int  x = static_cast<int>(t);
  if(blk_num_rows_ == 0) {
    memset(data_, x, sizeof(T)*num_rows_*stride_);
  } else {
    int32 mem_rows = blk_num_rows_ * row_blks_;
    memset(data_, x, sizeof(T) * mem_rows * stride_);
  }
}

template<typename T>
void CharacterMatrix<T>::Resize(MatrixIndexT rows, MatrixIndexT cols) {
  MatrixIndexT skip;
  MatrixIndexT real_cols;
  size_t size;
  void*   data;       // aligned memory block
  int32 mem_rows = rows;
  int32 mem_cols = cols;
  if (CharacterMatrixBase<T>::blk_num_rows_ > 0) {
    int32 x = cols % CharacterMatrixBase<T>::blk_num_cols_; // pad column first
    if (x > 0) {
      mem_cols += CharacterMatrixBase<T>::blk_num_cols_ - x;
    }
    x = rows % CharacterMatrixBase<T>::blk_num_rows_ ; // then we pad row
    if (x > 0) {
      mem_rows += CharacterMatrixBase<T>::blk_num_rows_ -x;
    }   
    CharacterMatrixBase<T>::col_blks_ = mem_cols / CharacterMatrixBase<T>::blk_num_cols_; 
    CharacterMatrixBase<T>::row_blks_ = mem_rows / CharacterMatrixBase<T>::blk_num_rows_;
  }
  // compute the size of skip and real cols
  skip = ((16 / sizeof(T)) - mem_cols % (16 / sizeof(T))) % (16 / sizeof(T));
  real_cols = mem_cols + skip; 
  size = static_cast<size_t>(mem_rows) * static_cast<size_t>(real_cols) * sizeof(T);
    
  // allocate the memory and set the right dimensions and parameters
  int pos_ret = posix_memalign(static_cast<void**>(&data), 16, size);
  //int pos_ret = posix_memalign(static_cast<void**>(data_), 16, size);
  if(pos_ret != 0) {
    KALDI_ERR << "Failed to do posix memory allot";
  }
  CharacterMatrixBase<T>::data_ = static_cast<T *> (data);
  CharacterMatrixBase<T>::num_rows_ = rows;
  CharacterMatrixBase<T>::num_cols_ = cols;
  CharacterMatrixBase<T>::stride_  = real_cols;
  this->SetZero();
}

template<typename T>
template<typename Real>
inline Real CharacterMatrixBase<T>::CharToReal(const T t) {
  int32 lower = std::numeric_limits<T>::min();
  Real x = static_cast<Real>(min_ + (t - lower) / increment_);
  return x;
}

template<typename T>
template<typename Real>
T CharacterMatrixBase<T>::RealToChar(const Real r) {
  int32 lower = std::numeric_limits<T>::min();
  T t = static_cast<T>((r - min_) * increment_ + lower);
  //if((r - min_) * increment_ + lower - t - 0.5)
// std::cout << " t before = "<<static_cast<int>(t)<<std::endl ;
// std::cout << " (r - min_) * increment_ + lower = " << (r - min_) * incremental_ + lower << std::endl ; 
 if(std::abs((r - min_) * increment_ + lower - static_cast<Real>(t)) > 0.5)  {
  if( (r - min_) * increment_ + lower < static_cast<Real>(0)) {
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
float CharacterMatrixBase<T>::T2R(const T &t) {
  int32 lower = std::numeric_limits<T>::min();
  float x = static_cast<float>(min_ + (t - lower) / increment_);
  return x;
}

template<typename T>
template<typename Real>
void CharacterMatrix<T>::CopyFromMat(const MatrixBase<Real> & M) {
  Resize(M.NumRows(),M.NumCols());
  //Real min = M.Min(), max = M.Max(); 
  Real Diff = static_cast<float>(0.05) * (M.Max() - M.Min());
  Real min = M.Min() - Diff, max = M.Max() + Diff;

  CharacterMatrixBase<T>::min_ = static_cast<float>(min);
  //std::cout << " min before = " << static_cast<float>(M.Min()) << " min after = " << static_cast<float> (min_) << std::endl ;
  int32 minChar = std::numeric_limits<T>::min(),
        maxChar = std::numeric_limits<T>::max();
  CharacterMatrixBase<T>::increment_ = static_cast<float>(static_cast<float>(maxChar - minChar)/(max - min));

  for (MatrixIndexT row = 0; row < M.NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M.NumCols(); col++) {
      (*this)(row, col) = CharacterMatrixBase<T>::template RealToChar<Real>(M(row, col));
    }
  }
}
template<typename Real>
template<typename OtherReal>
void CharacterMatrix<Real>::CopyFromMatrix(const CharacterMatrix<OtherReal> &M) {
  if ( sizeof(Real) == sizeof(OtherReal) && (void*)(&M) == (void*)this)
    return;
  Resize(M.NumRows(),M.NumCols());
  //KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
  int32 this_stride = CharacterMatrixBase<Real>::stride_, other_stride = M.Stride();
  Real *this_data = CharacterMatrixBase<Real>::data_;
  const OtherReal *other_data = M.Data();
  for (MatrixIndexT i = 0; i < CharacterMatrixBase<Real>::num_rows_; i++) 
    for(MatrixIndexT j = 0; j < CharacterMatrixBase<Real>::num_cols_; j++)
     // (*this_data)[i*this_stride + j] += (*other_data)[i * other_stride + j];
     (*this)(i,j) = M(i,j);
}
                                      
// Recover floating matrix  from char matrix
template<typename T>
template<typename Real>
void CharacterMatrixBase<T>::CopyToMat(Matrix<Real> *M) {
  M->Resize(num_rows_, num_cols_);
  for (MatrixIndexT row = 0; row < M->NumRows(); row++) {
    for (MatrixIndexT col = 0; col < M->NumCols(); col++) {
      (*M)(row, col) = CharToReal<Real>((*this)(row, col));
    }
  }
}
/*
template <typename T>
CharacterMatrix<T>& CharacterMatrix<T>::operator = (const CharacterMatrix& rhs) {
  if (CharacterMatrix<T>::NumRows() != rhs.NumRows() || CharacterMatrix<T>::NumCols() != rhs.NumCols()) {
    Resize(rhs.NumRows(), rhs.NumCols());
  }
  std::cout<<" we are here in operator= "<<std::endl ;
  CharacterMatrix<T>::CopyFromCharacterMatrix(rhs,"kNoTrans");
  return *this;
}
*/
template <typename T>
void CharacterMatrixBase<T>::CheckMatrix() {
  if(blk_num_rows_ != 0) {
    int32 row_num = row_blks_ * blk_num_rows_;
    int32 col_num = col_blks_ * blk_num_cols_;
    KALDI_ASSERT(row_num >= num_rows_);
    KALDI_ASSERT(col_num >= num_cols_);
    for(int32 row = 0; row < row_num; ++ row) {
      for(int32 col = num_cols_; col < col_num; ++ col ) {
        int32 x = static_cast<int32>( *(data_ + row * stride_ + col));
        KALDI_ASSERT (x == 0 ); 
      }
    }
    for (int32 row = num_rows_ ; row < row_num; ++ row ) {
      for (int32 col = 0; col < col_num; ++ col) {
        int32 x = static_cast<int32> (*(data_ +row * stride_ + col));
        KALDI_ASSERT(x == 0);
      }
    }
  }
}
template<typename Real>
class CharacterSubMatrix : public CharacterMatrixBase<Real> {

  public:
    CharacterSubMatrix(const CharacterMatrixBase<Real> &T,
                       const MatrixIndexT ro,
                       const MatrixIndexT r,
                       const MatrixIndexT col,
                       const MatrixIndexT c);
    ~CharacterSubMatrix() { }
    CharacterSubMatrix<Real> (const CharacterSubMatrix &other) :
        CharacterMatrixBase<Real> (other.data_, other.num_cols_, other.num_rows_,
             other.stride_) {}

  private:
    CharacterSubMatrix<Real> &operator = (const CharacterSubMatrix<Real> &other);
};

template<typename Real>
CharacterSubMatrix<Real>::CharacterSubMatrix(
    const CharacterMatrixBase<Real> &M,
    const MatrixIndexT ro,const MatrixIndexT r,
    const MatrixIndexT col, const MatrixIndexT c) {
  //KALDI_ASSERT();
  CharacterMatrixBase<Real>::num_rows_ = r;
  CharacterMatrixBase<Real>::num_cols_ = c;
  CharacterMatrixBase<Real>::stride_ = M.Stride();
  CharacterMatrixBase<Real>::data_ = M.Data_workaround()+ ro * M.Stride() + col;
}
template<typename Real>
CharacterSubMatrix<Real> CharacterMatrixBase<Real>::Range(
    const MatrixIndexT ro,const MatrixIndexT r, const MatrixIndexT col,
    MatrixIndexT c) {
  return CharacterSubMatrix<Real>(*this, ro, r, col, c);
}
                      

} //  namespace kaldi

#endif  // KALDI_CHARACTER_MATRIX_H_

