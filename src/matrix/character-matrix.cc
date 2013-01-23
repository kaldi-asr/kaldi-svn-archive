#include "character-matrix.h"
#include <iostream>
#include <smmintrin.h>//SSE4 intrinscis
namespace kaldi {

int DotProduct(unsigned char *x, signed char *y, MatrixIndexT length) {
  int32 sum=0;
  for(int32 i=0; i< length; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}
int DotProduct2(unsigned char *x, signed char *y, MatrixIndexT length) {
  int32 i, sum=0, a, b, c, d, e;
  for(i=0; i + 4 < length; i += 5) {
    a = x[i] * y[i];
    b = x[i+1] * y[i+1];
    c = x[i+2] * y[i+2];
    d = x[i+3] * y[i+3];
    e = x[i+4] * y[i+4];
    sum += a + b + c + d + e;
  }
  for(; i < length;  ++i)
   sum += x[i] * y[i];
  return sum;
}

int Sse4DotProduct(unsigned char *x, signed char *y, MatrixIndexT length) {
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

void Sse4DotProduct4fold1X4(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length){
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
  __m128  s1,s2,s3,s4;

  __m128i sum = _mm_setzero_si128();


  for (i=0; i+63<length; i+=64) {
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

  for (; i<length; i+=16) {
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

void Sse4DotProduct8fold1X4V3(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length){
    int i;
//    __m128i c, lo, hi;
//    __m128i *e, *f;
    __m128i c1, lo1, hi1;
    __m128i *e1, *f1;
    __m128i c2, lo2, hi2;
    __m128i *e2, *f2;
    __m128i c3, lo3, hi3;
    __m128i *e3, *f3;
    __m128i c4, lo4, hi4;
    __m128i *e4, *f4;
    __m128i sum = _mm_setzero_si128();

    __m128i sum11 = _mm_setzero_si128(); __m128i sum21 = _mm_setzero_si128(); __m128i sum31 = _mm_setzero_si128(); __m128i sum41 = _mm_setzero_si128();
    __m128i sum12 = _mm_setzero_si128(); __m128i sum22 = _mm_setzero_si128(); __m128i sum32 = _mm_setzero_si128(); __m128i sum42 = _mm_setzero_si128();
    __m128i sum13 = _mm_setzero_si128(); __m128i sum23 = _mm_setzero_si128(); __m128i sum33 = _mm_setzero_si128(); __m128i sum43 = _mm_setzero_si128();
    __m128i sum14 = _mm_setzero_si128(); __m128i sum24 = _mm_setzero_si128(); __m128i sum34 = _mm_setzero_si128(); __m128i sum44 = _mm_setzero_si128();

    __m128i sum1 = _mm_setzero_si128();
    __m128i sum2 = _mm_setzero_si128();
    __m128i sum3 = _mm_setzero_si128();
    __m128i sum4 = _mm_setzero_si128();
    __m128 s1,s2,s3,s4;

    for (i=0; i + 127< length; i+=128) {
        e1 = (__m128i*)(x+i);f1 = (__m128i*)(y1+i);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum11 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum11);
        e2 = (__m128i*)(x+i+16);f2 = (__m128i*)(y1+i+16);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum12 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum12);
        e3 = (__m128i*)(x+i+32);f3 = (__m128i*)(y1+i+32);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum13 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum13);
        e4 = (__m128i*)(x+i+48);f4 = (__m128i*)(y1+i+48);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum14 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum14);

        e1 = (__m128i*)(x+i+64);f1 = (__m128i*)(y1+i+64);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum11 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum11);
        e2 = (__m128i*)(x+i+80);f2 = (__m128i*)(y1+i+80);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum12 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum12);
        e3 = (__m128i*)(x+i+96);f3 = (__m128i*)(y1+i+96);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum13 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum13);
        e4 = (__m128i*)(x+i+112);f4 = (__m128i*)(y1+i+112);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum14 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum14);

        e1 = (__m128i*)(x+i);f1 = (__m128i*)(y2+i);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum21 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum21);
        e2 = (__m128i*)(x+i+16);f2 = (__m128i*)(y2+i+16);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum22 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum22);
        e3 = (__m128i*)(x+i+32);f3 = (__m128i*)(y2+i+32);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum23 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum23);
        e4 = (__m128i*)(x+i+48);f4 = (__m128i*)(y2+i+48);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum24 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum24);

        e1 = (__m128i*)(x+i+64);f1 = (__m128i*)(y2+i+64);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum21 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum21);
        e2 = (__m128i*)(x+i+80);f2 = (__m128i*)(y2+i+80);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum22 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum22);
        e3 = (__m128i*)(x+i+96);f3 = (__m128i*)(y2+i+96);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum23 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum23);
        e4 = (__m128i*)(x+i+112);f4 = (__m128i*)(y2+i+112);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum24 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum24);

        e1 = (__m128i*)(x+i);f1 = (__m128i*)(y3+i);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum31 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum31);
        e2 = (__m128i*)(x+i+16);f2 = (__m128i*)(y3+i+16);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum32 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum32);
        e3 = (__m128i*)(x+i+32);f3 = (__m128i*)(y3+i+32);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum33 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum33);
        e4 = (__m128i*)(x+i+48);f4 = (__m128i*)(y3+i+48);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum34 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum34);

        e1 = (__m128i*)(x+i+64);f1 = (__m128i*)(y3+i+64);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum31 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum31);
        e2 = (__m128i*)(x+i+80);f2 = (__m128i*)(y3+i+80);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum32 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum32);
        e3 = (__m128i*)(x+i+96);f3 = (__m128i*)(y3+i+96);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum33 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum33);
        e4 = (__m128i*)(x+i+112);f4 = (__m128i*)(y3+i+112);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum34 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum34);

        e1 = (__m128i*)(x+i);f1 = (__m128i*)(y4+i);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum41 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum41);
        e2 = (__m128i*)(x+i+16);f2 = (__m128i*)(y4+i+16);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum42 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum42);
        e3 = (__m128i*)(x+i+32);f3 = (__m128i*)(y4+i+32);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum43 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum43);
        e4 = (__m128i*)(x+i+48);f4 = (__m128i*)(y4+i+48);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum44 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum44);

        e1 = (__m128i*)(x+i+64);f1 = (__m128i*)(y4+i+64);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum41 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum41);
        e2 = (__m128i*)(x+i+80);f2 = (__m128i*)(y4+i+80);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum42 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum42);
        e3 = (__m128i*)(x+i+96);f3 = (__m128i*)(y4+i+96);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum43 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum43);
        e4 = (__m128i*)(x+i+112);f4 = (__m128i*)(y4+i+112);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum44 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum44);

    }

    sum1 = _mm_add_epi32(_mm_add_epi32(sum11, sum12), _mm_add_epi32(sum13, sum14));
    sum2 = _mm_add_epi32(_mm_add_epi32(sum21, sum22), _mm_add_epi32(sum23, sum24));
    sum3 = _mm_add_epi32(_mm_add_epi32(sum31, sum32), _mm_add_epi32(sum33, sum34));
    sum4 = _mm_add_epi32(_mm_add_epi32(sum41, sum42), _mm_add_epi32(sum43, sum44));


    for (; i<length; i+=16) {
        e1 = (__m128i*)(x+i);f1 = (__m128i*)(y1+i);c1 = _mm_maddubs_epi16(e1[0], f1[0]);lo1 = _mm_cvtepi16_epi32(c1);hi1 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c1, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo1, hi1), sum1);
        e2 = (__m128i*)(x+i);f2 = (__m128i*)(y2+i);c2 = _mm_maddubs_epi16(e2[0], f2[0]);lo2 = _mm_cvtepi16_epi32(c2);hi2 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c2, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo2, hi2), sum2);
        e3 = (__m128i*)(x+i);f3 = (__m128i*)(y3+i);c3 = _mm_maddubs_epi16(e3[0], f3[0]);lo3 = _mm_cvtepi16_epi32(c3);hi3 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c3, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo3, hi3), sum3);
        e4 = (__m128i*)(x+i);f4 = (__m128i*)(y4+i);c4 = _mm_maddubs_epi16(e4[0], f4[0]);lo4 = _mm_cvtepi16_epi32(c4);hi4 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c4, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo4, hi4), sum4);
    }

    s1 = _mm_castsi128_ps(sum1); s2 = _mm_castsi128_ps(sum2); s3 = _mm_castsi128_ps(sum3); s4 = _mm_castsi128_ps(sum4);
    _MM_TRANSPOSE4_PS(s1, s2, s3, s4);
    sum1 = _mm_castps_si128(s1); sum2 = _mm_castps_si128(s2); sum3 = _mm_castps_si128(s3); sum4 = _mm_castps_si128(s4);
    sum = _mm_add_epi32(_mm_add_epi32(sum1, sum2), _mm_add_epi32(sum3, sum4));

    result[0] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,2,1,0)));
    result[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,2,0,1)));
    result[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,0,1,2)));
    result[3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(0,2,1,3)));

}
 
void Sse4DotProduct8fold1X4V2(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length){
    int i;
    __m128i c, lo, hi;
    __m128i *e, *f;
    __m128i sum = _mm_setzero_si128();
    __m128i sum1 = _mm_setzero_si128();
    __m128i sum2 = _mm_setzero_si128();
    __m128i sum3 = _mm_setzero_si128();
    __m128i sum4 = _mm_setzero_si128();
    __m128 s1,s2,s3,s4;

    for (i=0; i + 127< length; i+=128) {
        e = (__m128i*)(x+i);f = (__m128i*)(y1+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+16);f = (__m128i*)(y1+i+16);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+32);f = (__m128i*)(y1+i+32);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+48);f = (__m128i*)(y1+i+48);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+64);f = (__m128i*)(y1+i+64);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+80);f = (__m128i*)(y1+i+80);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+96);f = (__m128i*)(y1+i+96);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i+112);f = (__m128i*)(y1+i+112);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i);f = (__m128i*)(y2+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+16);f = (__m128i*)(y2+i+16);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+32);f = (__m128i*)(y2+i+32);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+48);f = (__m128i*)(y2+i+48);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+64);f = (__m128i*)(y2+i+64);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+80);f = (__m128i*)(y2+i+80);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+96);f = (__m128i*)(y2+i+96);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i+112);f = (__m128i*)(y2+i+112);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i);f = (__m128i*)(y3+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+16);f = (__m128i*)(y3+i+16);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+32);f = (__m128i*)(y3+i+32);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+48);f = (__m128i*)(y3+i+48);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+64);f = (__m128i*)(y3+i+64);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+80);f = (__m128i*)(y3+i+80);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+96);f = (__m128i*)(y3+i+96);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i+112);f = (__m128i*)(y3+i+112);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i);f = (__m128i*)(y4+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+16);f = (__m128i*)(y4+i+16);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+32);f = (__m128i*)(y4+i+32);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+48);f = (__m128i*)(y4+i+48);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+64);f = (__m128i*)(y4+i+64);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+80);f = (__m128i*)(y4+i+80);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+96);f = (__m128i*)(y4+i+96);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
        e = (__m128i*)(x+i+112);f = (__m128i*)(y4+i+112);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);


    }
    for (; i< length; i+=16) {
        e = (__m128i*)(x+i);f = (__m128i*)(y1+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum1 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum1);
        e = (__m128i*)(x+i);f = (__m128i*)(y2+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum2 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum2);
        e = (__m128i*)(x+i);f = (__m128i*)(y3+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum3 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum3);
        e = (__m128i*)(x+i);f = (__m128i*)(y4+i);c = _mm_maddubs_epi16(e[0], f[0]);lo = _mm_cvtepi16_epi32(c);hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
        sum4 = _mm_add_epi32(_mm_add_epi32(lo, hi), sum4);
    }

    s1 = _mm_castsi128_ps(sum1); s2 = _mm_castsi128_ps(sum2); s3 = _mm_castsi128_ps(sum3); s4 = _mm_castsi128_ps(sum4);
    _MM_TRANSPOSE4_PS(s1, s2, s3, s4);
    sum1 = _mm_castps_si128(s1); sum2 = _mm_castps_si128(s2); sum3 = _mm_castps_si128(s3); sum4 = _mm_castps_si128(s4);
    sum = _mm_add_epi32(_mm_add_epi32(sum1, sum2), _mm_add_epi32(sum3, sum4));

    result[0] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,2,1,0)));
    result[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,2,0,1)));
    result[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(3,0,1,2)));
    result[3] = _mm_cvtsi128_si32(_mm_shuffle_epi32(sum, _MM_SHUFFLE(0,2,1,3)));

}

void Sse4DotProduct8fold1X4(unsigned char *x, signed char *y1, signed char *y2, signed char *y3, signed char *y4, int *result, MatrixIndexT length){
  int i;
  __m128i c11, c21, c31, c41, c12, c22, c32, c42, c13, c23, c33, c43, c14, c24, c34, c44;
  __m128i lo11, lo21, lo31, lo41, lo12, lo22, lo32, lo42, lo13, lo23, lo33, lo43, lo14, lo24, lo34, lo44;
  __m128i hi11, hi21, hi31, hi41, hi12, hi22, hi32, hi42, hi13, hi23, hi33, hi43, hi14, hi24, hi34, hi44;
  __m128i *f11, *f21, *f31, *f41, *f12, *f22, *f32, *f42, *f13, *f23, *f33, *f43, *f14, *f24, *f34, *f44;
        
  __m128i c15, c25, c35, c45, c16, c26, c36, c46, c17, c27, c37, c47, c18, c28, c38, c48;
  __m128i lo15, lo25, lo35, lo45, lo16, lo26, lo36, lo46, lo17, lo27, lo37, lo47, lo18, lo28, lo38, lo48;
  __m128i hi15, hi25, hi35, hi45, hi16, hi26, hi36, hi46, hi17, hi27, hi37, hi47, hi18, hi28, hi38, hi48;
  __m128i *f15, *f25, *f35, *f45, *f16, *f26, *f36, *f46, *f17, *f27, *f37, *f47, *f18, *f28, *f38, *f48;
        
  __m128i *e1, *e2, *e3, *e4, *e5, *e6, *e7, *e8;
        
  __m128i sum11 = _mm_setzero_si128(); __m128i sum21 = _mm_setzero_si128(); __m128i sum31 = _mm_setzero_si128(); __m128i sum41 = _mm_setzero_si128();
  __m128i sum12 = _mm_setzero_si128(); __m128i sum22 = _mm_setzero_si128(); __m128i sum32 = _mm_setzero_si128(); __m128i sum42 = _mm_setzero_si128();
  __m128i sum13 = _mm_setzero_si128(); __m128i sum23 = _mm_setzero_si128(); __m128i sum33 = _mm_setzero_si128(); __m128i sum43 = _mm_setzero_si128();
  __m128i sum14 = _mm_setzero_si128(); __m128i sum24 = _mm_setzero_si128(); __m128i sum34 = _mm_setzero_si128(); __m128i sum44 = _mm_setzero_si128();
  __m128i sum15 = _mm_setzero_si128(); __m128i sum25 = _mm_setzero_si128(); __m128i sum35 = _mm_setzero_si128(); __m128i sum45 = _mm_setzero_si128();
  __m128i sum16 = _mm_setzero_si128(); __m128i sum26 = _mm_setzero_si128(); __m128i sum36 = _mm_setzero_si128(); __m128i sum46 = _mm_setzero_si128();
  __m128i sum17 = _mm_setzero_si128(); __m128i sum27 = _mm_setzero_si128(); __m128i sum37 = _mm_setzero_si128(); __m128i sum47 = _mm_setzero_si128();
  __m128i sum18 = _mm_setzero_si128(); __m128i sum28 = _mm_setzero_si128(); __m128i sum38 = _mm_setzero_si128(); __m128i sum48 = _mm_setzero_si128();


  __m128i sum1 = _mm_setzero_si128();
  __m128i sum2 = _mm_setzero_si128();
  __m128i sum3 = _mm_setzero_si128();
  __m128i sum4 = _mm_setzero_si128();

  __m128  s1,s2,s3,s4;

  __m128i sum = _mm_setzero_si128();



  for (i=0; i+127< length; i+=128) {
    e1 = (__m128i*)(x+i); e2 = (__m128i*)(x+i+16); e3 = (__m128i*)(x+i+32); e4 = (__m128i*)(x+i+48);
    e5 = (__m128i*)(x+i+64); e6 = (__m128i*)(x+i+80); e7 = (__m128i*)(x+i+96); e8 = (__m128i*)(x+i+112);

    f11 = (__m128i*)(y1+i); f12 = (__m128i*)(y1+i+16); f13 = (__m128i*)(y1+i+32); f14 = (__m128i*)(y1+i+48);
    f21 = (__m128i*)(y2+i); f22 = (__m128i*)(y2+i+16); f23 = (__m128i*)(y2+i+32); f24 = (__m128i*)(y2+i+48);
    f31 = (__m128i*)(y3+i); f32 = (__m128i*)(y3+i+16); f33 = (__m128i*)(y3+i+32); f34 = (__m128i*)(y3+i+48);
    f41 = (__m128i*)(y4+i); f42 = (__m128i*)(y4+i+16); f43 = (__m128i*)(y4+i+32); f44 = (__m128i*)(y4+i+48);
    f15 = (__m128i*)(y1+i+64); f16 = (__m128i*)(y1+i+80); f17 = (__m128i*)(y1+i+96); f18 = (__m128i*)(y1+i+112);
    f25 = (__m128i*)(y2+i+64); f26 = (__m128i*)(y2+i+80); f27 = (__m128i*)(y2+i+96); f28 = (__m128i*)(y2+i+112);
    f35 = (__m128i*)(y3+i+64); f36 = (__m128i*)(y3+i+80); f37 = (__m128i*)(y3+i+96); f38 = (__m128i*)(y3+i+112);
    f45 = (__m128i*)(y4+i+64); f46 = (__m128i*)(y4+i+80); f47 = (__m128i*)(y4+i+96); f48 = (__m128i*)(y4+i+112);

    c11 = _mm_maddubs_epi16(e1[0], f11[0]); c12 = _mm_maddubs_epi16(e2[0], f12[0]); c13 = _mm_maddubs_epi16(e3[0], f13[0]); c14 = _mm_maddubs_epi16(e4[0], f14[0]);
    c21 = _mm_maddubs_epi16(e1[0], f21[0]); c22 = _mm_maddubs_epi16(e2[0], f22[0]); c23 = _mm_maddubs_epi16(e3[0], f23[0]); c24 = _mm_maddubs_epi16(e4[0], f24[0]);
    c31 = _mm_maddubs_epi16(e1[0], f31[0]); c32 = _mm_maddubs_epi16(e2[0], f32[0]); c33 = _mm_maddubs_epi16(e3[0], f33[0]); c34 = _mm_maddubs_epi16(e4[0], f34[0]);
    c41 = _mm_maddubs_epi16(e1[0], f41[0]); c42 = _mm_maddubs_epi16(e2[0], f42[0]); c43 = _mm_maddubs_epi16(e3[0], f43[0]); c44 = _mm_maddubs_epi16(e4[0], f44[0]);
    c15 = _mm_maddubs_epi16(e5[0], f15[0]); c16 = _mm_maddubs_epi16(e6[0], f16[0]); c17 = _mm_maddubs_epi16(e7[0], f17[0]); c18 = _mm_maddubs_epi16(e8[0], f18[0]);
    c25 = _mm_maddubs_epi16(e5[0], f25[0]); c26 = _mm_maddubs_epi16(e6[0], f26[0]); c27 = _mm_maddubs_epi16(e7[0], f27[0]); c28 = _mm_maddubs_epi16(e8[0], f28[0]);
    c35 = _mm_maddubs_epi16(e5[0], f35[0]); c36 = _mm_maddubs_epi16(e6[0], f36[0]); c37 = _mm_maddubs_epi16(e7[0], f37[0]); c38 = _mm_maddubs_epi16(e8[0], f38[0]);
    c45 = _mm_maddubs_epi16(e5[0], f45[0]); c46 = _mm_maddubs_epi16(e6[0], f46[0]); c47 = _mm_maddubs_epi16(e7[0], f47[0]); c48 = _mm_maddubs_epi16(e8[0], f48[0]);

    // unpack the 4 lowest 16-bit integers into 32 bits.
    lo11 = _mm_cvtepi16_epi32(c11); lo12 = _mm_cvtepi16_epi32(c12); lo13 = _mm_cvtepi16_epi32(c13); lo14 = _mm_cvtepi16_epi32(c14);
    lo21 = _mm_cvtepi16_epi32(c21); lo22 = _mm_cvtepi16_epi32(c22); lo23 = _mm_cvtepi16_epi32(c23); lo24 = _mm_cvtepi16_epi32(c24);
    lo31 = _mm_cvtepi16_epi32(c31); lo32 = _mm_cvtepi16_epi32(c32); lo33 = _mm_cvtepi16_epi32(c33); lo34 = _mm_cvtepi16_epi32(c34);
    lo41 = _mm_cvtepi16_epi32(c41); lo42 = _mm_cvtepi16_epi32(c42); lo43 = _mm_cvtepi16_epi32(c43); lo44 = _mm_cvtepi16_epi32(c44);
    lo15 = _mm_cvtepi16_epi32(c15); lo16 = _mm_cvtepi16_epi32(c16); lo17 = _mm_cvtepi16_epi32(c17); lo18 = _mm_cvtepi16_epi32(c18);
    lo25 = _mm_cvtepi16_epi32(c25); lo26 = _mm_cvtepi16_epi32(c26); lo27 = _mm_cvtepi16_epi32(c27); lo28 = _mm_cvtepi16_epi32(c28);
    lo35 = _mm_cvtepi16_epi32(c35); lo36 = _mm_cvtepi16_epi32(c36); lo37 = _mm_cvtepi16_epi32(c37); lo38 = _mm_cvtepi16_epi32(c38);
    lo45 = _mm_cvtepi16_epi32(c45); lo46 = _mm_cvtepi16_epi32(c46); lo47 = _mm_cvtepi16_epi32(c47); lo48 = _mm_cvtepi16_epi32(c48);

    // unpack the 4 highest 16-bit integers into 32 bits.
    hi11 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c11, 0x4e)); hi12 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c12, 0x4e)); hi13 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c13, 0x4e)); hi14 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c14, 0x4e));
    hi21 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c21, 0x4e)); hi22 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c22, 0x4e)); hi23 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c23, 0x4e)); hi24 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c24, 0x4e));
    hi31 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c31, 0x4e)); hi32 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c32, 0x4e)); hi33 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c33, 0x4e)); hi34 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c34, 0x4e));
    hi41 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c41, 0x4e)); hi42 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c42, 0x4e)); hi43 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c43, 0x4e)); hi44 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c44, 0x4e));
    hi15 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c15, 0x4e)); hi16 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c16, 0x4e)); hi17 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c17, 0x4e)); hi18 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c18, 0x4e));
    hi25 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c25, 0x4e)); hi26 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c26, 0x4e)); hi27 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c27, 0x4e)); hi28 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c28, 0x4e));
    hi35 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c35, 0x4e)); hi36 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c36, 0x4e)); hi37 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c37, 0x4e)); hi38 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c38, 0x4e));
    hi45 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c45, 0x4e)); hi46 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c46, 0x4e)); hi47 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c47, 0x4e)); hi48 = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c48, 0x4e));


    // pass the result to sum
    sum11 = _mm_add_epi32(_mm_add_epi32(lo11, hi11), sum11);  sum12 = _mm_add_epi32(_mm_add_epi32(lo12, hi12), sum12);  sum13 = _mm_add_epi32(_mm_add_epi32(lo13, hi13), sum13);  sum14 = _mm_add_epi32(_mm_add_epi32(lo14, hi14), sum14);
    sum21 = _mm_add_epi32(_mm_add_epi32(lo21, hi21), sum21);  sum22 = _mm_add_epi32(_mm_add_epi32(lo22, hi22), sum22);  sum23 = _mm_add_epi32(_mm_add_epi32(lo23, hi23), sum23);  sum24 = _mm_add_epi32(_mm_add_epi32(lo24, hi24), sum24);
    sum31 = _mm_add_epi32(_mm_add_epi32(lo31, hi31), sum31);  sum32 = _mm_add_epi32(_mm_add_epi32(lo32, hi32), sum32);  sum33 = _mm_add_epi32(_mm_add_epi32(lo33, hi33), sum33);  sum34 = _mm_add_epi32(_mm_add_epi32(lo34, hi34), sum34);
    sum41 = _mm_add_epi32(_mm_add_epi32(lo41, hi41), sum41);  sum42 = _mm_add_epi32(_mm_add_epi32(lo42, hi42), sum42);  sum43 = _mm_add_epi32(_mm_add_epi32(lo43, hi43), sum43);  sum44 = _mm_add_epi32(_mm_add_epi32(lo44, hi44), sum44);
    sum15 = _mm_add_epi32(_mm_add_epi32(lo15, hi15), sum15);  sum16 = _mm_add_epi32(_mm_add_epi32(lo16, hi16), sum16);  sum17 = _mm_add_epi32(_mm_add_epi32(lo17, hi17), sum17);  sum18 = _mm_add_epi32(_mm_add_epi32(lo18, hi18), sum18);
    sum25 = _mm_add_epi32(_mm_add_epi32(lo25, hi25), sum25);  sum26 = _mm_add_epi32(_mm_add_epi32(lo26, hi26), sum26);  sum27 = _mm_add_epi32(_mm_add_epi32(lo27, hi27), sum27);  sum28 = _mm_add_epi32(_mm_add_epi32(lo28, hi28), sum28);
    sum35 = _mm_add_epi32(_mm_add_epi32(lo35, hi35), sum35);  sum36 = _mm_add_epi32(_mm_add_epi32(lo36, hi36), sum36);  sum37 = _mm_add_epi32(_mm_add_epi32(lo37, hi37), sum37);  sum38 = _mm_add_epi32(_mm_add_epi32(lo38, hi38), sum38);
    sum45 = _mm_add_epi32(_mm_add_epi32(lo45, hi45), sum45);  sum46 = _mm_add_epi32(_mm_add_epi32(lo46, hi46), sum46);  sum47 = _mm_add_epi32(_mm_add_epi32(lo47, hi47), sum47);  sum48 = _mm_add_epi32(_mm_add_epi32(lo48, hi48), sum48);


  }
for (; i<length; i+=16) {
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

  sum1 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(sum11, sum12), _mm_add_epi32(sum13, sum14)), _mm_add_epi32(_mm_add_epi32(sum15, sum16), _mm_add_epi32(sum17, sum18)));
  sum2 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(sum21, sum22), _mm_add_epi32(sum23, sum24)), _mm_add_epi32(_mm_add_epi32(sum25, sum26), _mm_add_epi32(sum27, sum28)));
  sum3 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(sum31, sum32), _mm_add_epi32(sum33, sum34)), _mm_add_epi32(_mm_add_epi32(sum35, sum36), _mm_add_epi32(sum37, sum38)));
  sum4 = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(sum41, sum42), _mm_add_epi32(sum43, sum44)), _mm_add_epi32(_mm_add_epi32(sum45, sum46), _mm_add_epi32(sum47, sum48)));


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

int Sse4SumArray(unsigned char *x, MatrixIndexT length) {
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

} // namespace kaldi
