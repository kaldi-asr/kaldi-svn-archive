// util/kaldi-thread.h

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

#ifndef KALDI_THREAD_KALDI_THREAD_H_
#define KALDI_THREAD_KALDI_THREAD_H_ 1

#include <pthread.h>
#include "thread/kaldi-barrier.h"
// This header provides a convenient mechanism for parallelization.  The idea is
// that you have some range of integers, e.g. A ... B-1 (with B > A), and some
// function call that takes a range of integers, and you partition these up into
// a number of blocks.

// TODO: if needed, provide a workaround for Windows and other
// non-POSIX-compliant systems, possibly one that does not actually do
// multi-threading.


// Description of MultiThreadPool and it's usage:
//
// Usage of the RunMultiThreadedPersistent is the same as the usage of
// RunMultiThreaded, except that the object provided ust inherit MultiThreadable
// and it's run method isn't called, but operator() is called directly instead.
// Member variables num_threads_ and thread_id_ must NOT be redefined in the
// classes used, as they are called when using MultiThreadable*
//
// MultiThreadPool is a singleton class, it's instance is obtained using
// MultiThreadable::Instantiate(). First instantiation initializes the thread
// pool using g_num_threads threads, each of those threads runs infinite loop in
// ThreadWorker::run(). When RunMultiThreadedPersistent(c) is called, each
// ThreadWorker is given a pointer to a copy of c and calls c() in it's thread.
// After doing this, it Waits on barrier to sync with all the threads and the
// main one, then Waits again until it receives another job.

namespace kaldi {

extern int32 g_num_threads;  // Maximum number of threads (for programs that
// use threads, which is not many of them, e.g. the SGMM update program does.
// This is 8 by default.  You can change this on the command line, where
// used, with --num-threads.  Programs that think they will use threads
// should register it with their ParseOptions, as something like:
// po.Register("num-threads", &g_num_threads, "Number of threads to use.");

class MultiThreadable {
  // To create function that does part of the job, create class that inherits
  // this one, reimplements operator() and does part of the job based on
  //  thread_id_ and num_threads_
  // Note: example implementations are in thread/kaldi-thread-test.cc
 public:
  virtual void operator() () = 0;
  // Does the main function of the class
  //  Subclasses have to redefine this
  virtual ~MultiThreadable();
  // Optional destructor.  Note: the destructor
  // the object passed by the user will also be called, so
  // watch out.

 public:
  // do NOT reimplement thread_id_ and num_threads_ in derived classes, as they
  // are used in *MultiThreadable context, so the derived classes would not see
  // the changes made to them like this
  /* final */ int32 thread_id_;  // 0 <= thread_number < num_threads
  /* final */ int32 num_threads_;

 private:
  // Have additional member variables as needed.
};

class ThreadWorker {  // worker thread
  // handles waiting on barriers and launches it's job classes

 public:
  Barrier *barrier_;
  MultiThreadable *job_;

  static void *run(void *ThisPtr);
};

class ExampleClass {
 public:
  ExampleClass(const ExampleClass &other) {
    // .. optional initalizer.  Run sequentially;
    // initialized from object passed by user.
  }
  void operator() () {
    // Does the main function of the class
  }
  ~ExampleClass() {
    // Optional destructor.  Note: the destructor
    // the object passed by the user will also be called, so
    // watch out.
  }

  // This function should be provided. Give it this exact implementation, with
  // the class name replaced with your own class's name.
  static void *run(void *c_in) {
    ExampleClass *c = static_cast<ExampleClass*>(c_in);
    (*c)();  // call operator () on it.
    return NULL;
  }
 public:
  int32 thread_id_;  // 0 <= thread_number < num_threads
  int32 num_threads_;

 private:
  // Have additional member variables as needed.
};

template<class C> void RunMultiThreaded(const C &c_in) {
  KALDI_ASSERT(g_num_threads > 0);
  if (g_num_threads == 1) {  // Just run one copy.
    C c(c_in);  // create a copy of the object, just for consistency
    c.thread_id_ = 0;
    c.num_threads_ = 1;
    // with what happens in the multi-threaded case.
    C::run(&c);  // Note: this is the same as calling c(), but
    // we do it like this in case the user (ill-advisedly) put any
    // other statements in the static "run" function.
  } else {
    pthread_t *threads = new pthread_t[g_num_threads];
    std::vector<C> cvec(g_num_threads, c_in);  // all initialized with same
    // object.
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);
    for (int32 thread = 0; thread < g_num_threads; thread++) {
      cvec[thread].thread_id_ = thread;
      cvec[thread].num_threads_ = g_num_threads;
      int32 ret;
      if ((ret=pthread_create(&(threads[thread]),
                              &pthread_attr, C::run, &(cvec[thread])))) {
        const char *c = strerror(ret);
        if (c == NULL) { c = "[NULL]"; }
        KALDI_ERR << "Error creating thread, errno was: " << c;
      }
    }
    for (int32 thread = 0; thread < g_num_threads; thread++)
      if (pthread_join(threads[thread], NULL))
        KALDI_ERR << "Error rejoining thread.";
    delete [] threads;
  }
}

class TerminateThread: public MultiThreadable {  // job used to terminate thread
 public:
  TerminateThread() { }
  // Use default copy constructor and assignment operators.
  void operator() () {
    pthread_exit(NULL);
  }
};

class MultiThreadPool {  // singleton class for managing the thread pool
 public:
  static MultiThreadPool& Instantiate();

  void *run();
  void SetJobs(MultiThreadable** jobs);

 private:
  pthread_t *thread_ids_;
  std::vector<ThreadWorker> threads_;
  Barrier *barrier_;

  int32 num_threads_;
  static bool initialized_;

  void Initialize();  // this function creates the actual thread pool
  void Reinitialize();  // re-create the thread pool, used when g_num_threads is
  // changed
  void DeletePool();  // used by destructor and reinitialization - terminates
  // and deletes existing threas in thread pool
  ~MultiThreadPool();

 private:
  // prevent copying of existing instance etc.
  inline explicit MultiThreadPool() {}
  inline explicit MultiThreadPool(MultiThreadPool const&) {}
  inline MultiThreadPool& operator=(MultiThreadPool const&) { return *this; }
};

template<class C> void RunMultiThreadedPersistent(const C &c_in) {
  if (g_num_threads == 1) {
    C c(c_in);  // copy of c_in, for the consistency
    c.num_threads_ = 1;
    c.thread_id_ = 0;
    c();  // just call the method on object provided
  } else {
    MultiThreadPool::Instantiate();
    // we have to prepare jobs here, because it is the last place the class C is
    // known
    MultiThreadable** jobs = new MultiThreadable*[g_num_threads];
    for (int32 thread = 0; thread < g_num_threads; thread++) {
      jobs[thread] = new C(c_in);
    }
    MultiThreadPool::Instantiate().SetJobs(jobs);
    MultiThreadPool::Instantiate().run();

    for (int32 thread = 0; thread < g_num_threads; thread++) {
      delete jobs[thread];
    }
    delete [] jobs;
  }
}
}

#endif  // KALDI_THREAD_KALDI_THREAD_H_
