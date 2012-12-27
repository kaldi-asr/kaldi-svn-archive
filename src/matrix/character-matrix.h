#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <smmintrin.h>//SSE4 intrinscis

//a trial of Matrix class with SSE multiplication. By Xiao-hui Zhang 2012/12

// note from Dan: set your indent to 2 characters, not 4.
// note from Dan: I think you need this matrix class to contain
// floating-point (e.g. float) data members min_ and max_, which represent
// the values of the most negative and most positive values of type T.
// Or perhaps min_ and increment_ would make more sense, where increment_ is
// the amount we get from increasing the value of the character by one.
// Then we interpret the elements of the character matrix as integers in this
// way.  The operator () could possibly return float, then.
// You'd have to have your CopyFromMat function create these values appropriately.
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

using namespace std; // <-- Note from Dan: don't  do this, it's against the Google style.
// use std:: if you need something from std.

template<typename T>
class CharacterMatrix{
public:
    typedef T* iter;
    typedef const T* const_iter;
    typedef int32 MatrixIndexT;

    //constructors & destructor:
  CharacterMatrix() {create();} // note from Dan: this create() function is only
  // called once so put the code here unless you have other plans-- also, it should
  // have been called Init() if you had had it.
  
    // make it explicit to make statement like "vec<int> a = 10;" illegal.
   // no need for "explicit" if it takes >1 argument. [dan]
    explicit CharacterMatrix(MatrixIndexT r, MatrixIndexT c, const T& value = T()) { Resize(r, c value); }
    
    CharacterMatrix(const CharacterMatrix& m) { CopyFromMat(m); } // copy constructor
    ~CharacterMatrix() { //  cout<<"destructor called"<<endl;
      free(data_); // [dan]: what happens if data_ = NULL?
    }

  
    //operator overloading functions:
    
    CharacterMatrix& operator = (const CharacterMatrix&); // assignment operator

    T&  operator() (MatrixIndexT r, MatrixIndexT c) {
        return *(data_ + r * stride_ + c);
    }
    const  T&  operator() (MatrixIndexT r, MatrixIndexT c) const {
        return *(data_ + r * stride_ + c);
    }

  // [dan]:  I don't think we should have  these iterators.
    //other public member functions:
    iter begin() { return data_; }
    const_iter begin() const { return data_; }
    
    inline MatrixIndexT NumRows() const { return num_rows_; }
    inline MatrixIndexT NumCols() const { return num_rows_; }
    inline MatrixIndexT NumRealCols() const { return stride_; }

    // [dan]: delete clear() and empty().  We can use Resize(0, 0).
    void clear() {  free(data_); }
    bool empty() const { return num_rows_ == 0 || num_cols_ == 0; }
    void SetZero();
    void Set(T value);
    
    void Resize(MatrixIndexT, MatrixIndexT, const T&);
    void CopyFromMat(const CharacterMatrix<T> & M);
    void Transpose(const CharacterMatrix<T> & M);
    void MatMat(const CharacterMatrix<T> & M1, const CharacterMatrix<T> & M2);
private:
    iter  data_;
    MatrixIndexT num_cols_;
    MatrixIndexT num_rows_;
    MatrixIndexT stride_;
  // from Dan: if you need this function it should be called Sse4DotProduct.
  // but it probably doesn't belong here, e.g. could be a static inline function
  // declared and defined in character-matrix.cc.
    short SSE4_dot_product(unsigned char *x, signed char *y, MatrixIndexT length);
};

template <typename T>
void CharacterMatrix<T>::create()
{
    data_ = 0;
    num_rows_ = 0;
    num_cols_ = 0;
    stride_  = 0;
}

    
template<typename T>
void CharacterMatrix<T>::SetZero() {
    if (num_cols_ == stride_)
        memset(data_, 0, sizeof(T)*num_rows_*num_cols_);
    else
        for (MatrixIndexT row = 0; row < num_rows_; row++)
            memset(data_ + row*stride_, 0, sizeof(T)*num_cols_);
}

template<typename T>
void CharacterMatrix<T>::Set(T value) {
    for (MatrixIndexT row = 0; row < num_rows_; row++) {
         for (MatrixIndexT col = 0; col < num_cols_; col++) {
               (*this)(row, col) = value;
             }
    }
}

template <typename T>
void vec<T>::Resize(MatrixIndexT rows, MatrixIndexT cols, const T& value)
{
    MatrixIndexT skip;
    MatrixIndexT real_cols;
    size_t size;
    void*   data;       // aligned memory block
    void*   free_data;  // memory block to be really freed
    
    // compute the size of skip and real cols
    skip = ((16 / sizeof(T)) - cols % (16 / sizeof(T))) % (16 / sizeof(T));
    real_cols = cols + skip;
    size = static_cast<size_t>(rows) * static_cast<size_t>(real_cols) * sizeof(Real);
    
    // allocate the memory and set the right dimensions and parameters
    if (posix_memalign((void**)&x, 16, size*sizeof(T)) == 0 ) {
        data_ = static_cast<T *> (data);
    } // else what?  KALDI_ERROR? [dan]
    num_rows_ = rows;
    num_cols_ = cols;
    stride_  = real_cols;
    this->Set(value);
}

template<typename T>
void CharacterMatrix<T>::CopyFromCharacterMatrix(const CharacterMatrix<T> & M) {
    assert(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
    for (MatrixIndexT row = 0; row < num_rows_; row++) {
        for (MatrixIndexT col = 0; col < num_cols_; col++) {
            (*this)(row, col) = M(row, col);
        }
    }
}

template<typename T>
void CharacterMatrix<T>::Transpose(const CharacterMatrix<T> & M){
    (*this).Resize(M.NumCols(), M.NumRows(), 0);
    for (MatrixIndexT row = 0; row < M.NumCols(); row++) {
        for (MatrixIndexT col = 0; col < M.NumRows(); col++) {
            (*this)(row, col) = m(col, row);
        }
    }
}


template<typename T>
void CharacterMatrix<T>::MatMat(const CharacterMatrix<T> & M1, const CharacterMatrix<T> & M2){
    assert(M1.NumCols == M2.NumRows());
    (*this).Resize(M1.NumRows(), M2.NumCols(), 0);
    CharacterMatrix<T> M2T;
    M2T.Transpose(M2);
    short tmp;
    for (MatrixIndexT row = 0; row < M1.NumRows(); row++) {
        for (MatrixIndexT col = 0; col < M2.NumCols(); col++) {
            tmp = SSE4_dot_product((unsigned char *)(M1.begin() + row * M1.NumRealCols(),(signed char *)(M2T.begin() + col * M2T.NumRealCols(), M1.NumCols());
            (*this)(row, col) = static_cast<T> (tmp);
        }
    }
}

template <typename T>
CharacterMatrix<T>& vec<T>::operator = (const CharacterMatrix& rhs) {
    if (CharacterMatrix<T>::NumRows() != rhs.NumRows() || CharacterMatrix<T>::NumCols() != rhs.NumCols()) {
        Resize(rhs.NumRows(), rhs.NumCols());
    }
    CharacterMatrix<T>::CopyFromMat(rhs);
    return *this;
}

template<typename T>
short CharacterMatrix<T>::SSE4_dot_product(unsigned char *x, signed char *y, MatrixIndexT length)
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
