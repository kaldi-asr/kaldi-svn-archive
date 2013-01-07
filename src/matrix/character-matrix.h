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

inline void Sse4DotProduct1X4(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length){
  int i;
  __m128i c11, c21, c31, c41, c12, c22, c32, c42, c13, c23, c33, c43, c14, c24, c34, c44;
  __m128i lo11, lo21, lo31, lo41, lo12, lo22, lo32, lo42, lo13, lo23, lo33, lo43, lo14, lo24, lo34, lo44;
  __m128i hi11, hi21, hi31, hi41, hi12, hi22, hi32, hi42, hi13, hi23, hi33, hi43, hi14, hi24, hi34, hi44;
  __m128i *f11, *f21, *f31, *f41, *f12, *f22, *f32, *f42, *f13, *f23, *f33, *f43, *f14, *f24, *f34, *f44;
  __m128i *e1, *e2, *e3, *e4;
  __m128i sum11 = _mm_setzero_si128(); __m128i sum21 = _mm_setzero_si128(); __m128i sum31 = _mm_setzero_si128(); __m128i sum41 = _mm_setzero_si128();
  __m128i sum12 = _mm_setzero_si128(); __m128i sum22 = _mm_setzero_si128(); __m128i sum32 = _mm_setzero_si128(); __m128i sum42 = _mm_setzero_si128();
  __m128i sum13 = _mm_setzero_si128(); __m128i sum23 = _mm_setzero_si128(); __m128i sum33 = _mm_setzero_si128(); __m128i sum43 = _mm_setzero_si128();
  __m128i sum14 = _mm_setzero_si128(); __m128i sum24 = _mm_setzero_si128(); __m128i sum34 = _mm_setzero_si128(); __m128i sum44 = _mm_setzero_si128();
    
  __m128i sum1 = _mm_setzero_si128();
  __m128i sum2 = _mm_setzero_si128();
  __m128i sum3 = _mm_setzero_si128();
  __m128i sum4 = _mm_setzero_si128();
  __m128 s1,s2,s3,s4;

  __m128i sum = _mm_setzero_si128();


  for (i=0; i+63 < length; i+=64) {
    e1 = (__m128i*)(x+i); e2 = (__m128i*)(x+i+16); e3 = (__m128i*)(x+i+32); e4 = (__m128i*)(x+i+48);

    f11 = (__m128i*)(y1+i); f12 = (__m128i*)(y1+i+16); f13 = (__m128i*)(y1+i+32); f14 = (__m128i*)(y1+i+48);
    f21 = (__m128i*)(y2+i); f22 = (__m128i*)(y2+i+16); f23 = (__m128i*)(y2+i+32); f24 = (__m128i*)(y2+i+48);
    f31 = (__m128i*)(y3+i); f32 = (__m128i*)(y3+i+16); f33 = (__m128i*)(y3+i+32); f34 = (__m128i*)(y3+i+48);
    f41 = (__m128i*)(y4+i); f42 = (__m128i*)(y4+i+16); f43 = (__m128i*)(y4+i+32); f44 = (__m128i*)(y4+i+48);

    c11 = _mm_maddubs_epi16(e1[0], f11[0]); c12 = _mm_maddubs_epi16(e2[0], f12[0]); c13 = _mm_maddubs_epi16(e3[0], f13[0]); c14 = _mm_maddubs_epi16(e4[0], f14[0]);
    c21 = _mm_maddubs_epi16(e1[0], f21[0]); c22 = _mm_maddubs_epi16(e2[0], f22[0]); c23 = _mm_maddubs_epi16(e3[0], f23[0]); c24 = _mm_maddubs_epi16(e4[0], f24[0]);
    c31 = _mm_maddubs_epi16(e1[0], f31[0]); c32 = _mm_maddubs_epi16(e2[0], f32[0]); c33 = _mm_maddubs_epi16(e3[0], f33[0]); c34 = _mm_maddubs_epi16(e4[0], f34[0]);
    c41 = _mm_maddubs_epi16(e1[0], f41[0]); c42 = _mm_maddubs_epi16(e2[0], f42[0]); c43 = _mm_maddubs_epi16(e3[0], f43[0]); c44 = _mm_maddubs_epi16(e4[0], f44[0]);

    // unpack the 4 lowest 16-bit integers into 32 bits.
    lo11 = _mm_cvtepi16_epi32(c11); lo12 = _mm_cvtepi16_epi32(c12); lo13 = _mm_cvtepi16_epi32(c13); lo14 = _mm_cvtepi16_epi32(c14);
    lo21 = _mm_cvtepi16_epi32(c21); lo22 = _mm_cvtepi16_epi32(c22); lo23 = _mm_cvtepi16_epi32(c23); lo24 = _mm_cvtepi16_epi32(c24);
    lo31 = _mm_cvtepi16_epi32(c31); lo32 = _mm_cvtepi16_epi32(c32); lo33 = _mm_cvtepi16_epi32(c33); lo34 = _mm_cvtepi16_epi32(c34);
    lo41 = _mm_cvtepi16_epi32(c41); lo42 = _mm_cvtepi16_epi32(c42); lo43 = _mm_cvtepi16_epi32(c43); lo44 = _mm_cvtepi16_epi32(c44);

    // unpack the 4 highest 16-bit integers into 32 bits.
    hi11 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c11, 0x4e)); hi12 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c12, 0x4e)); hi13 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c13, 0x4e)); hi14 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c14, 0x4e));
    hi21 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c21, 0x4e)); hi22 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c22, 0x4e)); hi23 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c23, 0x4e)); hi24 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c24, 0x4e));
    hi31 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c31, 0x4e)); hi32 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c32, 0x4e)); hi33 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c33, 0x4e)); hi34 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c34, 0x4e));
    hi41 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c41, 0x4e)); hi42 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c42, 0x4e)); hi43 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c43, 0x4e)); hi44 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c44, 0x4e));

    // pass the result to sum
    sum11 = _mm_add_epi32(_mm_add_epi32(lo11, hi11), sum11);  sum12 = _mm_add_epi32(_mm_add_epi32(lo12, hi12), sum12);  sum13 = _mm_add_epi32(_mm_add_epi32(lo13, hi13), sum13);  sum14 = _mm_add_epi32(_mm_add_epi32(lo14, hi14), sum14);
    sum21 = _mm_add_epi32(_mm_add_epi32(lo21, hi21), sum21);  sum22 = _mm_add_epi32(_mm_add_epi32(lo22, hi22), sum22);  sum23 = _mm_add_epi32(_mm_add_epi32(lo23, hi23), sum23);  sum24 = _mm_add_epi32(_mm_add_epi32(lo24, hi24), sum24);
    sum31 = _mm_add_epi32(_mm_add_epi32(lo31, hi31), sum31);  sum32 = _mm_add_epi32(_mm_add_epi32(lo32, hi32), sum32);  sum33 = _mm_add_epi32(_mm_add_epi32(lo33, hi33), sum33);  sum34 = _mm_add_epi32(_mm_add_epi32(lo34, hi34), sum34);
    sum41 = _mm_add_epi32(_mm_add_epi32(lo41, hi41), sum41);  sum42 = _mm_add_epi32(_mm_add_epi32(lo42, hi42), sum42);  sum43 = _mm_add_epi32(_mm_add_epi32(lo43, hi43), sum43);  sum44 = _mm_add_epi32(_mm_add_epi32(lo44, hi44), sum44);

  }
    
  for (; i < length; i+=16) {
    e1 = (__m128i*)(x+i);
    f11 = (__m128i*)(y1+i);
    f21 = (__m128i*)(y2+i);
    f31 = (__m128i*)(y3+i);
    f41 = (__m128i*)(y4+i); 
    
    c11 = _mm_maddubs_epi16(e1[0], f11[0]);
    c21 = _mm_maddubs_epi16(e1[0], f21[0]);
    c31 = _mm_maddubs_epi16(e1[0], f31[0]);
    c41 = _mm_maddubs_epi16(e1[0], f41[0]);
    
    // unpack the 4 lowest 16-bit integers into 32 bits.
    lo11 = _mm_cvtepi16_epi32(c11);
    lo21 = _mm_cvtepi16_epi32(c21); 
    lo31 = _mm_cvtepi16_epi32(c31);
    lo41 = _mm_cvtepi16_epi32(c41);
    
    // unpack the 4 highest 16-bit integers into 32 bits.
    hi11 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c11, 0x4e));
    hi21 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c21, 0x4e));
    hi31 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c31, 0x4e));
    hi41 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c41, 0x4e));
    
    // pass the result to sum
    sum11 = _mm_add_epi32(_mm_add_epi32(lo11, hi11), sum11);
    sum21 = _mm_add_epi32(_mm_add_epi32(lo21, hi21), sum21);
    sum31 = _mm_add_epi32(_mm_add_epi32(lo31, hi31), sum31);
    sum41 = _mm_add_epi32(_mm_add_epi32(lo41, hi41), sum41);
  }
    
  sum1 = _mm_add_epi32(_mm_add_epi32(sum11, sum12), _mm_add_epi32(sum13, sum14));
  sum2 = _mm_add_epi32(_mm_add_epi32(sum21, sum22), _mm_add_epi32(sum23, sum24));
  sum3 = _mm_add_epi32(_mm_add_epi32(sum31, sum32), _mm_add_epi32(sum33, sum34));
  sum4 = _mm_add_epi32(_mm_add_epi32(sum41, sum42), _mm_add_epi32(sum43, sum44));

//    sum = _mm_hadd_epi32(sum,sum); // perform horizontal addition to sum up the partial dot-products
//    sum = _mm_hadd_epi32(sum,sum); // perform horizontal addition to sum up the partial dot-products
  
//    sum1 = _mm_hadd_epi32(sum1,sum1);
//    sum1 = _mm_hadd_epi32(sum1,sum1);
//    sum2 = _mm_hadd_epi32(sum2,sum2);
//    sum2 = _mm_hadd_epi32(sum2,sum2);
//    sum3 = _mm_hadd_epi32(sum3,sum3);
//    sum3 = _mm_hadd_epi32(sum3,sum3);
//    sum4 = _mm_hadd_epi32(sum4,sum4);
//    sum4 = _mm_hadd_epi32(sum4,sum4);
//    result[0] = _mm_cvtsi128_si32(sum1);
//    result[1] = _mm_cvtsi128_si32(sum2);
//    result[2] = _mm_cvtsi128_si32(sum3);
//    result[3] = _mm_cvtsi128_si32(sum4);

    
  s1 = _mm_castsi128_ps(sum1); s2 = _mm_castsi128_ps(sum2); s3 = _mm_castsi128_ps(sum3); s4 = _mm_castsi128_ps(sum4);
  _MM_TRANSPOSE4_PS(s1, s2, s3, s4);
  sum1 = _mm_castps_si128(s1); sum2 = _mm_castps_si128(s2); sum3 = _mm_castps_si128(s3); sum4 = _mm_castps_si128(s4);
  sum = _mm_add_epi32(_mm_add_epi32(sum1, sum2), _mm_add_epi32(sum3, sum4));
    
    
  result[0] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,2,1,0)));
  result[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,2,0,1)));
  result[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,0,1,2)));
  result[3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(0,2,1,3)));
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
  //Real min = M.Min(), max = M.Max(); 
  Real Diff = static_cast<float>(0.05) * (M.Max() - M.Min());
  Real min = M.Min() - Diff, max = M.Max() + Diff;

  min_ = static_cast<float>(min);
  //std::cout << " min before = " << static_cast<float>(M.Min()) << " min after = " << static_cast<float> (min_) << std::endl ;
  int32 minChar = std::numeric_limits<T>::min(),
        maxChar = std::numeric_limits<T>::max();
  incremental_ = static_cast<float>(static_cast<float>(maxChar - minChar)/(max - min));

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

