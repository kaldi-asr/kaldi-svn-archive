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
#include "thread/kaldi-semaphore.h"
#include "matrix/kaldi-matrix.h"
namespace kaldi {

template<typename Real>
class MyThreadClass2 {  // Doing Matrix multiplication.
 public:
  MyThreadClass2(CharacterMatrix<unsigned char> &M1, CharacterMatrix<signed char> &M2, 
                         int32 row_num,int32 col_num, Matrix<Real> *i): row_num_(row_num),
                                               col_num_(col_num),iptr_(i)
                                               {private_counter_.Resize(row_num_, col_num_);
                                                M1_.CopyFromMatrix(M1);
                                                M2_.CopyFromMatrix(M2);
                                               }
  // Use default copy constructor and assignment operators.
  void operator() () {
    int32 block_size = (row_num_+ (num_threads_-1) ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(row_num_, start + block_size);
  //  for (int32 row = start; row < end; row++)
    private_counter_.AddVecMat(1,M1_,kNoTrans, M2_,kTrans, 0 , start, end);
 // }
   }
  ~MyThreadClass2() {
    
   (*iptr_).AddMat(1,private_counter_,kNoTrans);
  }

  static void *run(void *c_in) {
    MyThreadClass2 *c = static_cast<MyThreadClass2*>(c_in);
    (*c)();  // call operator () on it.
    return NULL;
  }

 public:
  int32 thread_id_;  // 0 <= thread_number < num_threads
  int32 num_threads_;

 private:
  MyThreadClass2() { }  // Disallow empty constructor.
  int32 row_num_;
  int32 col_num_;
  Matrix<Real> *iptr_;
  Matrix<Real> private_counter_;
  CharacterMatrix<unsigned char> M1_;
  CharacterMatrix<signed char> M2_;
};

struct thread_test_struct { // start + end of integers to sum up.
  int32 start;
  int32 end;
  Semaphore empty_semaphore_;
  Semaphore full_semaphore_;
  bool done_;
  thread_test_struct()
  {
    start = 0;
    end = 0;
    empty_semaphore_ = 1;
    full_semaphore_ = 1;
    done_ = false;
  }
};

void* sum_ints(void* input) {
  thread_test_struct *ptr = static_cast<thread_test_struct*>(input);
  size_t ans = 0;
  for (int32 i = ptr->start; i < ptr->end; i++)
    ans += i;
  return (void*) ans;
}
void ExampleDone(pthread_t  thread_, void** iptr) {
     if( pthread_join(thread_, iptr))
            perror("error: Error rejoining thread.");
}

// Example Test :
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
     //std::cout << " start and end on thread " << thread <<" = " << c[thread].start << " " << c[thread].end << std::endl ;
     if ((ret = pthread_create(&(threads_[thread]), 
                               &pthread_attr, sum_ints ,&(c[thread])))) {
       perror("error:");
       KALDI_ERR << "Could not creare a new thread";      
      }
      c[thread].empty_semaphore_.Wait();
      c[thread].full_semaphore_.Signal();
   }
  //std::cout <<" empty val = " << c[thread].empty_semaphore_.GetValue() << std::endl; 
  for( int32 thread = 0; thread < num_thread; thread++) {
  //  if ( pthread_join(threads_[thread],&iptr_ ))
  //    KALDI_ERR << "Error rejoining thread.";
      ExampleDone((threads_[thread]), &iptr_);
      c[thread].empty_semaphore_.Signal();
      c[thread].full_semaphore_.Wait();
      c[thread].done_ = true;
      tot += reinterpret_cast<size_t>(iptr_);
  }
  //std::cout <<" empty val = " << empty_semaphore_.GetValue() << std::endl;
  KALDI_LOG << " total is " << tot;
  delete [] threads_;
}
  
template<typename Real>
void TestThreads2(int32 num_threads) {
 g_num_threads = num_threads ;
 int32 row_num = 10;
 int32 col = 10 ;
 int32 row2 = 10;
 Matrix<Real> Mr1(row_num, col);
 Mr1.SetRandn();
 CharacterMatrix<unsigned char> M1;
 M1.CopyFromMat(Mr1);

 Matrix<Real> Mr2(row2, col);
 Mr2.SetRandn();
 CharacterMatrix<signed char> M2;
 M2.CopyFromMat(Mr2);

 Matrix<Real> tot(row_num, row2);
 MyThreadClass2<Real> c(M1,M2,row_num,row2, &tot);
 RunMultiThreaded(c);
 
 } 


// End of Test;
}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  //TestThreadsSimple();
   TestThreads2<float>(10);
}

