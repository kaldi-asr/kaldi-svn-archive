// Base class for a thread-able class, containing wrappers for threads, mutex, and condition variables for ease of usage in thread pool, mainly with reference to the following paper and its example codes:
// Implementation and Usage of a ThreadPool based on POSIX Threads by Ronald Kriemann, Max-Planck-Institut für Mathematik, Technical Report no.2, 2003
// Xiao-hui Zhang


#include "base/kaldi-error.h"
#include "base/kaldi-common.h"
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sched.h>
#include <cmath>
#include "threadbase.h"

namespace kaldi{
namespace ThreadPool{
    
// routine to call Thread::run() method
extern "C" void *RunThread (void *arg){
  if (arg != NULL){
    ((Thread*) arg)->Run();
    ((Thread*) arg)->ResetRunning();
  }
  return NULL;
}

// constructor and destructor
Thread::Thread (const int athread_no):running_( false ), thread_no_(athread_no){}

Thread::~Thread() {
    // request cancellation of the thread if running
    if (running_) Cancel();
}
// access local data

void Thread::SetThreadNo (const int no){ thread_no_ = no; }

// thread management

// create thread (actually start it)
void Thread::Create(const bool detached, const bool sscope){
    if (!running_){
      int ret;
      pthread_attr_t thread_attr;
      if ((ret = pthread_attr_init( & thread_attr )) != 0){
        KALDI_ERR << "Thread create : pthread_attr_init failed.";
        return;
      }
      if(detached){
      // detache created thread from calling thread
        if ((ret = pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_DETACHED )) != 0){
          KALDI_ERR << "Thread create : pthread_attr_setdetachstate failed.";
          return;
        }
      }
      if(sscope){
      // use system-wide scheduling for thread
        if ((ret = pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM )) != 0 ){
          KALDI_ERR << "Thread create : pthread_attr_setscope failed";
          return;
        }
      }
      if ((ret = pthread_create(&thread_id_, &thread_attr, RunThread, this ) != 0))
        KALDI_ERR << "Thread create : pthread_create failed.";
      else
        running_ = true;
      // remove attribute
      pthread_attr_destroy(&thread_attr);
    }
    else
      KALDI_LOG << "Thread create : thread is already running.";
}

// detach thread
void Thread::Detach(){
  if(running_){
    int ret;
    // detach thread
    if ((ret = pthread_detach(thread_id_)) != 0)
      KALDI_ERR << "Thread detach : pthread_detach failed.";
    }
}

// synchronise with thread (wait until finished)
void Thread::Join(){
  if (running_){
    int ret;
    // wait for thread to finish
    if ((ret = pthread_join( thread_id_, NULL )) != 0)
      KALDI_ERR << "Thread join : pthread_join failed.";
    running_ = false;
  }
}

// request cancellation of thread
void Thread::Cancel(){
  if (running_){
    int ret;
    if ((ret = pthread_cancel(thread_id_)) != 0)
      KALDI_ERR << "Thread cancel : pthread_cancel failed.";
  }
}

// functions to be called by a thread itself
// terminate thread
void Thread::Exit(){
  if (running_ && (pthread_self() == thread_id_)){
    void *ret_val = NULL;
    pthread_exit(ret_val);
    running_ = false;
  }
}

// put thread to sleep (milli + nano seconds)
void Thread::Sleep (const double sec){
  if (running_){
    struct timespec interval;
    if (sec <= 0.0){
      interval.tv_sec  = 0;
      interval.tv_nsec = 0;
    }
    else{
      interval.tv_sec  = time_t( std::floor( sec ) );
      interval.tv_nsec = long( (sec - interval.tv_sec) * 1e6 );
    }
    nanosleep(&interval, 0);
  }
}

}// namespace TreadPool
}// namespace kaldi
