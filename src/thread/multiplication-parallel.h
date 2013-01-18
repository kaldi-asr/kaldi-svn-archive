//Hi
// util/kaldi-thread-test3.cc

// Copyright 2012  Johns Hopkins University (Author: Pegah Ghahremani)


#include "base/kaldi-common.h"
#include "thread/kaldi-thread.h"
#include "thread/kaldi-semaphore.h"
#include "matrix/kaldi-matrix.h"
namespace kaldi {

template<typename Real>
class MultiplicationParallel {  // Doing Matrix multiplication.
 public:
  MultiplicationParallel(CharacterMatrix<unsigned char> &M1, CharacterMatrix<signed char> &M2,  
                                          MatrixBase<Real> *i): row_num_(M1.NumRows()),
                                          col_num_(M2.NumRows()),iptr_(i)
                                               { 
                                                private_counter_.Resize(row_num_, col_num_);
                                                M1_ = &M1;
                                                (*M1_).CopyFromMatrix(M1);
                                                M2_ = &M2;
                                                (*M2_).CopyFromMatrix(M2);
                                               }
  // Use default copy constructor and assignment operators.
  void operator() () {
    int32 block_size = (row_num_ ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(row_num_, start + block_size);
    private_counter_.AddVecMat(1, M1_, kNoTrans, M2_, kTrans, 0 , start, end);
   }

  ~MultiplicationParallel() {
   (*iptr_).AddMat(1,private_counter_,kNoTrans);
  }
  static void *run(void *c_in) {
    MultiplicationParallel *c = static_cast<MultiplicationParallel*>(c_in);
    (*c)();  // call operator () on it.
    return NULL;
  }

 public:
  int32 thread_id_;  // 0 <= thread_number < num_threads
  int32 num_threads_;

 private:
  MultiplicationParallel() { }  // Disallow empty constructor.
  int32 row_num_;
  int32 col_num_;
  MatrixBase<Real> *iptr_;
  Matrix<Real> private_counter_;
  CharacterMatrix<unsigned char> *M1_;
  CharacterMatrix<signed char> *M2_ ;
};

}
