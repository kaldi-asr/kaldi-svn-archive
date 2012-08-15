// util/kaldi-thread-test.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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



namespace kaldi {

class MyThreadClass { // Sums up integers from 0 to max_to_count-1.
 public:
  MyThreadClass(int32 max_to_count, int32 *i): max_to_count_(max_to_count),
                                               iptr_(i),
                                               private_counter_(0) { }
  // Use default copy constructor and assignment operators.
  void operator () () {
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
    (*c)(); // call operator () on it.
    return NULL;
  }  
  
 public:
  int32 thread_id_; // 0 <= thread_number < num_threads
  int32 num_threads_;
  
 private:
  MyThreadClass() { };  // Disallow empty constructor.
  int32 max_to_count_;
  int32 *iptr_;
  int32 private_counter_;
};

class MyThreadClassPersist: public MultiThreadable { // variant of previous class for testing of persistent threads
 public:
  MyThreadClassPersist(int32 max_to_count, int32 *i): max_to_count_(max_to_count),
                                               iptr_(i),
                                               private_counter_(0) { }
  // Use default copy constructor and assignment operators.
  void operator () () {
    int32 block_size = (max_to_count_+ (num_threads_-1) ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(max_to_count_, start + block_size);
    for (int32 j = start; j < end; j++)
      private_counter_ += j;
  }
  ~MyThreadClassPersist() {
    *iptr_ += private_counter_;
  }
  
 public:
  // do NOT redefine thread_id_ and num_threads_, as they are used in
  // *MultiThreadable context
  
 private:
  MyThreadClassPersist() { };  // Disallow empty constructor.
  int32 max_to_count_;
  int32 *iptr_;
  int32 private_counter_;
};

class MyThreadClassPersist2: public MultiThreadable { // another version, just
  // to be able to check if jobs are exchanged properly
 public:
  int modif_; // one more variable to make this class of different size
  // (because different sizes of object were problem in early implementation,
  // so it's good to check this)
  MyThreadClassPersist2(int32 max_to_count, int32 *i): modif_(-1),
                                                max_to_count_(max_to_count),
                                                iptr_(i),
                                                private_counter_(0)
  { }
  // Use default copy constructor and assignment operators.
  void operator () () {
    int32 block_size = (max_to_count_+ (num_threads_-1) ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(max_to_count_, start + block_size);
    for (int32 j = start; j < end; j++)
      private_counter_ += j*modif_; // just count downwards
  }
  ~MyThreadClassPersist2() {
    *iptr_ += private_counter_;
  }
  
 public:
  // do NOT redefine thread_id_ and num_threads_, as they are used in
  // *MultiThreadable context
  
 private:
  MyThreadClassPersist2() { };  // Disallow empty constructor.
  int32 max_to_count_;
  int32 *iptr_;
  int32 private_counter_;
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
  // now check if the solution with persistent threads works as well
  // for starter's just one thread
  g_num_threads = 1;
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  g_num_threads = 8;
  // let's check if the thread pool works, so we have to set g_num_threads>1
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  // let's try the same thing again, to see if the worker objects get reset
  // properly
  // Note: again, just uncomment following line for simple benchmarking
  // for(int i=0; i<100000; i++)
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  // now let's try another jobs to see if exchanging works
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist2 c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == -(10000*(10000-1))/2);
  }
  g_num_threads = 2;
  // let's try to modify the number of threads
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  g_num_threads = 1;
  // once again only one thread
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  g_num_threads = 8;
  // back to 8 threads
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClassPersist c(max_to_count, &tot);
    RunMultiThreadedPersistent(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  g_num_threads = 8;
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  TestThreads();
}

