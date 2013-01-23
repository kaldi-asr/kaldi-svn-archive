// Class for thread pool, mainly with reference to the following paper and its example codes:
// Implementation and Usage of a ThreadPool based on POSIX Threads by Ronald Kriemann, Max-Planck-Institut für Mathematik, Technical Report no.2, 2003
// Xiao-hui Zhang

#ifndef THREADPOOL_H_
#define THREADPOOL_H_ 1 

#include "base/kaldi-error.h"
#include "base/kaldi-common.h"
#include "threadbase.h"
#include <list>
namespace kaldi {
namespace ThreadPool {
    
// forward decl. for internal class
class TPoolThr;
// implements a thread pool, e.g. takes jobs and
// executes them in threads

class TPool{
  friend class TPoolThr;
  public:
  // class for a job in the pool
  

protected:
// maximum degree of parallelism
unsigned int  max_parallel_;

// array of threads, handled by pool
TPoolThr **threads_;

// list of idle threads
std::list<TPoolThr *> idle_threads_;

// condition for synchronisation of idle list
Condition idle_cond_;

public:
class Job;

// construct thread pool with max_p threads
TPool ( const unsigned int max_p );

// wait for all threads to finish and destruct thread pool
~TPool ();

// return number of internal threads, e.g. maximal parallel degree
unsigned int MaxParallel() const { return max_parallel_; }


// run, stop and synch with job

// enqueue job in thread pool, e.g. execute job by the first freed thread
// ptr is an optional argument passed to the "run" method of job, contating the parameters needed for the job
// if del is true, the job object will be deleted after finishing "Run"
void Run(Job *job, void *ptr = NULL, const bool del = false );

// synchronise with \a job, i.e. wait until finished
void Sync(Job *job);

// synchronise with all running jobs
void SyncAll();
    
protected:
// manage pool threads
// return idle thread from pool
TPoolThr *GetIdle();

// insert idle thread into pool
void AppendIdle ( TPoolThr * t );
};
// to access the global thread-pool
// thread handled by threadpool

class TPool::Job{
 protected:
  // number of processor this job was assigned to
  const int  job_no_;

  // mutex for synchronisation
  Mutex sync_mutex_;
    
 public:
  // construct job object with n as job number
  Job (const int n = -1 ):job_no_(n){}

  // destruct job object
  virtual ~Job (){
  if (sync_mutex_.IsLocked())
    KALDI_LOG << "(Job) destructor : job is still running!";
  }
    
  // method to be executed by the thread (actual work should be here!)
  virtual void Run ( void * ptr ) = 0 ;
    
  // return assigned job number
  int JobNo() const { return job_no_; }

  // lock the internal mutex
  void Lock() { sync_mutex_.Lock(); }

  // unlock internal mutex
  void Unlock() { sync_mutex_.Unlock(); }
};



class TPoolThr : public Thread{
  protected:
    // pool we are in  
    TPool *        pool_;
          
    // job to run and data for it
    TPool::Job *  job_;
    void *         data_ptr_;
    
    // should the job be deleted upon completion
    bool           del_job_;
    
    // condition for job-waiting
    Condition   work_cond_;
      
    // indicates end-of-thread
    bool           end_;
    
    // mutex for preventing premature deletion
    Mutex         del_mutex_;
    
  public:
    TPoolThr ( const int n, TPool * p )
            : Thread(n), pool_(p), job_(NULL), data_ptr_(NULL), del_job_(false), end_(false){}
    ~TPoolThr() {}
      
    // parallel running method
    void Run();

    // set the job with optional data
    void SetJob(TPool::Job * j, void * p, const bool del );
    
    // give access to delete mutex

    Mutex & DelMutex() {
      return del_mutex_;
    }

    // quit thread (reset data and wake up)
    void Quit();
};
 

// init global thread_pool with max_p threads
void  Init( const unsigned int max_p );

// run a job in global thread pool with a ptr passed to job->run()
void  Run(TPool::Job *job, void *ptr = NULL, const bool del = false );

// synchronise with \a job
void  Sync(TPool::Job *job );

// synchronise with all jobs
void  SyncAll();

// finish global thread pool
void  Done();
   
}// namespace ThreadPool
}// namespace kaldi

#endif  // THREADPOOL_H_

