// Base class for a thread-able class, containing wrappers for threads, mutex, and condition variables for ease of usage in thread pool, mainly with reference to the following paper and its example codes:
// Implementation and Usage of a ThreadPool based on POSIX Threads by Ronald Kriemann, Max-Planck-Institut für Mathematik, Technical Report no.2, 2003
// Xiao-hui Zhang

#ifndef Thread_H_
#define Thread_H_ 1

#include <cstdio>
#include <pthread.h>
namespace kaldi{
namespace ThreadPool{
   
//  defines basic thread interface
class Thread{
  protected:
    pthread_t  thread_id_;
    bool       running_;  // is the thread running or not
    int        thread_no_;  // no of thread
        
  public:
    // constructor and destructor
    // construct thread with thread number thread_no
    Thread ( const int thread_no = -1 );

    // destruct thread, if thread is running, it will be canceled
    virtual ~Thread ();

    // access local data

    // return thread number
    int ThreadNo() const { return thread_no_; }

    // set thread number to n
    void SetThreadNo ( const int  n );
   
    // actual method to be executed by thread
    virtual void Run () = 0;
   
    // thread management
  
    // create thread (actually start it);
    // if detached is true, the thread will be started in detached mode,
    // e.g. can not be joined.
    // if sscope is true, the thread is started in system scope, e.g.
    // the thread competes for resources with all other threads of all
    // processes on the system; if sscope is false, the competition is
    // only process local
    void Create ( const bool  detached = false,
                  const bool  sscope   = false );
    // detach thread
    void Detach ();
    
    // synchronise with thread (wait until finished)
    void Join   ();

    // request cancellation of thread
    void Cancel ();

  protected:
    // functions to be called by a thread itself 
    // terminate thread
    void Exit   ();

    // put thread to sleep for <sec> seconds
    void Sleep  ( const double sec );

  public:
    // resets running-status (used in _run_proc, see Thread.cc)
    void ResetRunning() { running_ = false; }
    
};

    // wrapper for pthread_mutex, which is similar to kaldi-mutex.h and .cc
class Mutex{
  protected:
    // the mutex itself and the mutex-attr
    pthread_mutex_t      mutex_;
    pthread_mutexattr_t  mutex_attr_;
    
  public:
    // constructor and destructor
    Mutex(){
      pthread_mutexattr_init(&mutex_attr_);
      pthread_mutex_init(&mutex_, &mutex_attr_);
    }
    ~Mutex(){
      pthread_mutex_destroy(&mutex_);
      pthread_mutexattr_destroy(&mutex_attr_);
    }

    // usual behavior of a mutex
    // lock mutex
    void  Lock() { pthread_mutex_lock(&mutex_); }

    // unlock mutex
    void  Unlock() { pthread_mutex_unlock(&mutex_); }

    // return true if mutex is locked and false, otherwise
    bool IsLocked() {
      if (pthread_mutex_trylock(&mutex_) != 0)
        return true;
      else
        return false;
    }
};
    
// wrapper for condition variable, derived from mutex to
// allow locking of condition to inspect or modify the predicate

class Condition : public Mutex {
  private:
    // our condition variable
    pthread_cond_t cond_;

  public:

    // constructor and destructor
    Condition() { pthread_cond_init(&cond_, NULL); }
    ~Condition() { pthread_cond_destroy(&cond_); }

    // condition variable related methods
    // wait for signal to arrive
    void Wait() { pthread_cond_wait(&cond_, &mutex_); }

    // restart one of the threads, waiting on the cond. variable
    void Signal() { pthread_cond_signal(&cond_ ); }

    // restart all waiting threads
    void Broadcast() { pthread_cond_broadcast(&cond_); }
};

}// namespace ThereadPool
}// namespace kaldi

#endif  // Thread_H_
