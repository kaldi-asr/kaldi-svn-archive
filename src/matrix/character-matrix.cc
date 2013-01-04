#include "character-matrix.h"
#include <iostream>
namespace kaldi {

template<typename T>
short CharacterMatrix<T>::Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length)
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
