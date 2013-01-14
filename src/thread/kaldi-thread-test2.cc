// util/kaldi-thread-test.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
//                 Frantisek Skala

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "thread/kaldi-thread.h"
#include "matrix/kaldi-matrix.h"
namespace kaldi {

/*
class MyThreadClass {  // Sums up integers from 0 to max_to_count-1.
 public:
  MyThreadClass(int32 max_to_count, int32 *i): max_to_count_(max_to_count),
                                               iptr_(i),
                                               private_counter_(0) { }
  // Use default copy constructor and assignment operators.
  void operator() () {
    int32 block_size = (max_to_count_+ (num_threads_-1) ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(max_to_count_, start + block_size);
    for (int32 j = start; j < end; j++)
      private_counter_ += j;
  }
  ~MyThreadClass() {
    *iptr_ += private_counter_;
  }

  static void *run(void *c_in) {
    MyThreadClass *c = static_cast<MyThreadClass*>(c_in);
    (*c)();  // call operator () on it.
    return NULL;
  }

 public:
  int32 thread_id_;  // 0 <= thread_number < num_threads
  int32 num_threads_;

 private:
  MyThreadClass() { }  // Disallow empty constructor.
  int32 max_to_count_;
  int32 *iptr_;
  int32 private_counter_;
};

template<typename Real>
template<class Mat>
class MyThreadClass2 {  // Doing Matrix multiplication.
 public:
  MyThreadClass2(Mat<Real> &M1, Mat<Real> &M2, int32 row_num, Mat<Real> *i): row_num_(row_num),
                                               iptr_(i),
                                               private_counter_(Mat<Real> c1(M1.row, M2.col)) { }
  // Use default copy constructor and assignment operators.
  void operator() () {
    int32 block_size = (row_num_+ (num_threads_-1) ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(row_num_, start + block_size);
  //  for (int32 row = start; row < end; row++)
    private_counter_.AddVecMat(1,M1,kNoTrans, M2,kTrans, 0 , start, end);
 // }
  ~MyThreadClass() {
    *iptr_ += private_counter_;
  }

  static void *run(void *c_in) {
    MyThreadClass *c = static_cast<MyThreadClass*>(c_in);
    (*c)();  // call operator () on it.
    return NULL;
  }

 public:
  int32 thread_id_;  // 0 <= thread_number < num_threads
  int32 num_threads_;

 private:
  MyThreadClass() { }  // Disallow empty constructor.
  int32 row_num_;
  Mat<Real> *iptr_;
  Mat<Real> private_counter_;
};
  
void TestThreads() {
  g_num_threads = 8;
  // run method with temporary threads on 8 threads
  // Note: uncomment following line for the possibility of simple benchmarking
  // for(int i=0; i<100000; i++)
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClass c(max_to_count, &tot);
    RunMultiThreaded(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  g_num_threads = 1;
  // let's try the same, but with only one thread
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClass c(max_to_count, &tot);
    RunMultiThreaded(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
}
template<Real>
void TestThreads2(int32 num_threads) {
 g_num_threads = num_threads ;
 int32 row_num = 64;
 int32 col = 10 ;
 int32 row2 = 100;
 Matrix<Real> Mr1(row_num, col);
 Mr1.SetRandn();
 CharacterMatrix<unsigned char> M1;
 M1.CopyFromMat(Mr1);

 Matrix<Real> Mr2(row2, col);
 Mr2.SetRandn();
 CharacterMatrix<signed char> M2;
 M2.CopyFromMat(Mr2);

 CharacterMatrix<Real> tot(row_num, col);
 MyThreadClass2<CharacterMatrix> c(M1,M2,row_num, &tot);
 RunMultiThreaded(c);
  
 } */


struct thread_test_struct { // start + end of integers to sum up.
  int32 start;
  int32 end;
};

void* sum_ints(void* input) {
  thread_test_struct *ptr = static_cast<thread_test_struct*>(input);
  size_t ans = 0;
  for (int32 i = ptr->start; i < ptr->end; i++)
    ans += i;
  return (void*) ans;
}


void TestThreadsSimple() {
  // test code here that creates an array of the structs,
  // then, multiple times, spawns threads, then joins them
  // and sums up the resulting integers (cast back from void* to size_t).
  /*
  pthread_attr_t 
    pthread_create
    pthread_t
     pthread_join
  */
  int32 num_thread = 10;
  int32 max_to_count = 100;
  int32 tot = 0 ;
  void *iptr_;
  
  int32 block_size = (max_to_count + (num_thread - 1))/ num_thread;
  //std::vector<thread_test_struct> c;
  thread_test_struct *c = new thread_test_struct[num_thread];
  pthread_attr_t pthread_attr;
  pthread_attr_init(&pthread_attr);
  pthread_t *threads_ = new pthread_t[num_thread];
  //threads_(new pthread_t[g_num_threads]);
  //std::vector<pthread_t> threads_;
  for( int32 thread = 0; thread < num_thread; thread++) {
     int32 ret;
     c[thread].start = block_size * thread;
     c[thread].end = std::min(max_to_count, c[thread].start + block_size);
     std::cout << " start and end on thread " << thread <<" = " << c[thread].start << " " << c[thread].end << std::endl ;
     if ((ret = pthread_create(&(threads_[thread]), 
                               &pthread_attr, sum_ints ,&(c[thread])))) {
       perror("error:");
       KALDI_ERR << "Could not creare a new thread";      
      }
   }
  for( int32 thread = 0; thread < num_thread; thread++) {
    if ( pthread_join(threads_[thread],&iptr_ ))
      KALDI_ERR << "Error rejoining thread.";
      tot += reinterpret_cast<size_t>(iptr_);
  }
  KALDI_LOG << " total is " << tot;
  delete [] threads_;
}

  
}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  //TestThreads();
  TestThreadsSimple();
  //TestThreads<float>(num_threads);
}

