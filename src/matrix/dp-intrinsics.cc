// This code sample demonstrates how to use C, SSE3, SSSE3 and SSE4 instrinsics to calculate the dot product of two 16X1 vectors, by Xiao-hui Zhang.

// Reference:
//[1]. Intel® C++ Compiler for Linux* Intrinsics Reference. Document number: 312482-001US
//[2]. Improving the speed of neural networks on CPUs, Vincent Vanhoucke, Andrew W. Senior, Mark W. Mao.,
//     Deep Learning and Unsupervised Feature Learning Workshop, NIPS 2011
//[3]. Intel® C++ Intrinsics Reference. Document Number: 312482-002US

//If you compile this code using GCC, please make sure that your complier version is equal to or higher than 4.3, otherwise SSSE3 and SSE4 intrinsics are not supported. Also, to enable SSE extensions, you might have to do g++ like this: g++ -o t intr.cpp -msse4. When needed, please refer to other details at http://gcc.gnu.org/onlinedocs/gcc/i386-and-x86_002d64-Options.html

#include <stdio.h>
#include <pmmintrin.h> //SSE3 intrinscis
#include <tmmintrin.h>//SSSE3 intrinscis
#include <smmintrin.h>//SSE4 intrinscis
#include <stdlib.h>
//double *foo(void) {
//    double *var;//create array of size 10
//    int     ok;
//    
//    ok = posix_memalign((void**)&var, 64, 10*sizeof(double));
//    
//    if(ok != 0)
//        return NULL;
//    
//    return var;
//}

#define SIZE 16

//Computes dot product using C
float dot_product(float *a, float *b);

//Computes dot product using SSE2 intrinsics
float SSE3_dot_product(float *a, float *b);

//Computes dot product using SSSE3 intrinsics
short SSSE3_dot_product(unsigned char *a, signed char *b);

//Computes dot product using SSE4 intrinsics
short SSE4_dot_product(unsigned char *a, signed char *b);

int main()
{
    unsigned char *x;
    int ok;
    int i;
    ok = posix_memalign((void**)&x, 64, 16*sizeof(unsigned char));
    if(ok == 0){
        for (i=0; i<16; i++) {
            x[i]=200+i;
        }
    }
    signed char *y;
    ok = posix_memalign((void**)&y, 64, 16*sizeof(signed char));
    if(ok == 0){
        for (i=0; i<16; i++) {
            y[i]=1-2*(i%2);
        }
    }
    
    float a[] = {201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216};
    float b[] = {1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1};
    
//    signed char y[] = {1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1};


    float product;
    float sse3_product;
    short ssse3_product;
    short sse4_product;

    product= dot_product(a, b);
    printf("Dot Product computed by C: %f\n", product);
    
    sse3_product =SSE3_dot_product(a, b);
    printf("Dot Product computed by SSE3 intrinsics:  %f\n", sse3_product);
    
    ssse3_product =SSSE3_dot_product(x,y);
    printf("Dot Product computed by SSSE3 intrinsics:  %d\n", ssse3_product);
    
    sse4_product =SSE4_dot_product(x,y);
    printf("Dot Product computed by SSE4 intrinsics:  %d\n", sse4_product);

    return 0; }

float dot_product(float *a, float *b)
{
    int i;
    int sum=0;
    for(i=0; i<SIZE; i++)
    {
        sum += a[i]*b[i];
    }
    return sum; }

float SSE3_dot_product(float *a, float *b)
{
    float result;
    int i;
    __m128 pa, pb, sum, c;
    sum = _mm_setzero_ps();  //sets sum to zero
    
    for(i=0; i<SIZE; i+=4)
    {
        pa = _mm_loadu_ps(a+i);//loads unaligned sub-array a[i]~a[i+3] into pa
        pb = _mm_loadu_ps(b+i);//loads unaligned sub-array b[i]~b[i+3] into pb
        c = _mm_mul_ps(pa, pb); //performs multiplication to get partial dot-products
        c = _mm_hadd_ps(c, c); //performs horizontal addition to sum up the partial dot-products
        sum = _mm_add_ps(sum, c);  //performs vertical addition
    }
    sum = _mm_hadd_ps(sum, sum);
    _mm_store_ss(&result, sum);
    return result;
}

short SSSE3_dot_product(unsigned char *x, signed char *y)
{
    __m128i a, b;
    __m128i *e;
    __m128i sum = _mm_setzero_si128();
    short result;
    e = (__m128i*)x;
    a = e[0];
    e = (__m128i*)y;
    b = e[0];
    
    __m128i c = _mm_maddubs_epi16(a, b); // performs dot-product in 2X2 blocks
    // unpack the 4 lowest 16-bit integers into 32 bits.
    __m128i lo = _mm_srai_epi32(_mm_unpacklo_epi16(c, c),16);
    // unpack the 4 highest 16-bit integers into 32 bits.
    __m128i hi = _mm_srai_epi32(_mm_unpackhi_epi16(c, c),16);
    
    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);  // pass the result to sum
    sum = _mm_hadd_epi32(sum,sum); // perform vertical addition to sum up the partial dot-products
    sum = _mm_hadd_epi32(sum,sum); // perform vertical addition to sum up the partial dot-products
    
    result = _mm_cvtsi128_si32(sum); // extract dot-product result by moving the least significant 32 bits of the 128-bit "sum" to a 32-bit integer result
      
    return result;
}

short SSE4_dot_product(unsigned char *x, signed char *y)
{
    __m128i a, b;
    __m128i *e;
    __m128i sum = _mm_setzero_si128();
    short result;
    e = (__m128i*)x;
    a = e[0];
    e = (__m128i*)y;
    b = e[0];
    
    __m128i c = _mm_maddubs_epi16(a, b); // performs dot-product in 2X2 blocks
    
    // unpack the 4 lowest 16-bit integers into 32 bits.
    __m128i lo = _mm_cvtepi16_epi32(c);
    // unpack the 4 highest 16-bit integers into 32 bits.
    __m128i hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
    
    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);  // pass the result to sum
    sum = _mm_hadd_epi32(sum,sum); // perform vertical addition to sum up the partial dot-products
    sum = _mm_hadd_epi32(sum,sum); // perform vertical addition to sum up the partial dot-products
    
    result = _mm_cvtsi128_si32(sum); // extract dot-product result by moving the least significant 32 bits of the 128-bit "sum" to a 32-bit integer result
    
    return result;
}

