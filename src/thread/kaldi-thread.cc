// util/kaldi-thread.cc

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
int32 g_num_threads = 8;  // Initialize this global variable.

MultiThreadable::~MultiThreadable() {
  // default implementation does nothing
}

void *ThreadWorker::run(void *this_ptr_void) {
  ThreadWorker *this_ptr = static_cast<ThreadWorker*>(this_ptr_void);
  while (true) {
    this_ptr->barrier_->Wait();  // wait until parent comes with workers
    (*(this_ptr->job_))();  // call operator () on worker.
    this_ptr->barrier_->Wait();  // work is done, sync with parent
  }
  return NULL;
}

bool MultiThreadPool::initialized_ = false;
MultiThreadPool& MultiThreadPool::Instantiate() {
  static MultiThreadPool instance;  // Note: static variable inside function
  // gets initialized automatically after first function call
  if (initialized_ == true && instance.num_threads_ != g_num_threads) {
    instance.Reinitialize();
  }
  if (initialized_ == false) {
    instance.Initialize();
  }
  return instance;
}

void MultiThreadPool::Reinitialize() {  // re-create thread pool, used only if
  // g_num_threads was changed

  if (this->num_threads_ > 1) {
    // if the thread pool was actually created
    // substitute all jobs with the ones that terminate the threads
    MultiThreadable** jobs = new MultiThreadable*[this->num_threads_];
    for (int32 thread = 0; thread < this->num_threads_; thread++) {
      jobs[thread] = new TerminateThread();
      this->threads_[thread].job_ = (jobs[thread]);
    }
    this->barrier_->Wait();  // wake up the threads, so they terminate

    for (int32 thread = 0; thread < this->num_threads_; thread++) {
      if (pthread_join(this->thread_ids_[thread], NULL)) {  // wait for it's
        // termination
        KALDI_ERR << "Error rejoining thread.";
      }
    }

    for (int32 thread = 0; thread < this->num_threads_; thread++) {
      delete jobs[thread];
    }
    delete [] jobs;
    delete [] this->thread_ids_;
    delete this->barrier_;
  }
  this->initialized_ = false;
}

void MultiThreadPool::Initialize() {  // this function creates the actual thread
  // pool
  KALDI_ASSERT(g_num_threads > 0);  // number of threads must be at least one
  KALDI_ASSERT(this->initialized_ == false);  // cannot be initialized more than
  // once

  this->num_threads_ = g_num_threads;

  if (g_num_threads == 1) {
    // there is nothing we have to do...
  } else {
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);

    this->threads_ = std::vector<ThreadWorker>(g_num_threads);
    this->thread_ids_ = new pthread_t[g_num_threads];

    this->barrier_ = new Barrier(g_num_threads+1);  // barrier for all the
    // threads + parent

    for (int32 thread = 0; thread < g_num_threads; thread++) {
      this->threads_[thread].barrier_ = barrier_;
      int32 ret;

      if ((ret=pthread_create(&(this->thread_ids_[thread]),
                              &pthread_attr,
                              ThreadWorker::run,
                              &(this->threads_[thread])))) {
        const char *c = strerror(ret);
        if (c == NULL) { c = "[NULL]"; }
        KALDI_ERR << "Error creating thread, errno was: " << c;
      }
    }
  }

  this->initialized_ = true;
}

void *MultiThreadPool::run() {
  this->barrier_->Wait();  // wake up the jobs
  this->barrier_->Wait();  // wait until all of them finish
  return NULL;
}

void MultiThreadPool::SetJobs(MultiThreadable** jobs) {
  for (int32 thread = 0; thread < g_num_threads; thread++) {
    this->threads_[thread].job_ = (jobs[thread]);
    this->threads_[thread].job_->thread_id_ = thread;
    this->threads_[thread].job_->num_threads_ = g_num_threads;
  }
}

}  // end namespace kaldi
