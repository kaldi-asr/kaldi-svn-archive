// Class for thread pool, mainly with reference to the following paper and its example codes:
// Implementation and Usage of a ThreadPool based on POSIX Threads by Ronald Kriemann, Max-Planck-Institut für Mathematik, Technical Report no.2, 2003
// Xiao-hui Zhang

#include <pthread.h>
#include "threadpool.h"

namespace kaldi{
namespace ThreadPool{
    
namespace{
// global thread-pool
TPool *thread_pool = NULL;
}// namespace anonymous, equals to "static"
    
// ThreadPool - implementation
TPool::TPool ( const unsigned int  max_p ){
  // create max_p threads for pool
  max_parallel_ = max_p;
  threads_ = new TPoolThr*[max_parallel_];
  if (threads_ == NULL){
    max_parallel_ = 0;
    KALDI_LOG << "(TPool) TPool : could not allocate thread array";
  }
  for (unsigned int  i = 0; i < max_parallel_; i++){
    threads_[i] = new TPoolThr(i, this);
    KALDI_LOG << i << " th thread created.";
    if ( threads_ == NULL )
      KALDI_LOG << "(TPool) TPool : could not allocate thread";
    else
      threads_[i]->Create(true, true);
  }
}

TPool::~TPool (){
  KALDI_LOG << "Will destroy the thread pool";
  // wait till all threads have finished
  SyncAll();
  // finish all thread
  KALDI_LOG << "Sycronization of all threads finished. Will quit them";
  for ( unsigned int  i = 0; i < max_parallel_; i++ )
    threads_[i]->Quit();
  KALDI_LOG << "All threads quited successfully.";
  // cancel and delete all threads (not really safe !)
  for ( unsigned int  i = 0; i < max_parallel_; i++ ){
    threads_[i]->DelMutex().Lock();
    delete threads_[i];
  }
  delete[] threads_;
}


// run, stop and synch with job
void TPool::Run(TPool::Job *job, void *ptr, const bool del){
  if ( job == NULL )
    return;
  // run in parallel thread
  TPoolThr *thr = GetIdle();
  // lock job for synchronisation
  job->Lock();
  // attach job to thread
  thr->SetJob(job, ptr, del);
  KALDI_LOG << "Job "<< job->JobNo()<<" is assigned to an idle thread.";
}

// wait until <job> was executed
void TPool::Sync(Job * job){
  if (job == NULL)
    return;
  job->Lock();
  job->Unlock();
}

// wait until all jobs have been executed
void TPool::SyncAll (){
  while (true){
    idle_cond_.Lock();
    // wait until next thread becomes idle
    KALDI_LOG << "Synconizing all threads; Num. idle threads: " << idle_threads_.size() <<" Num. tot. threads: " << max_parallel_;
    if ( idle_threads_.size() < max_parallel_ )
      idle_cond_.Wait();
    else{ break; }
    idle_cond_.Unlock();
  }
}

// manage pool threads
// return idle thread form pool
TPoolThr *TPool::GetIdle(){
  while ( true ) {
    // wait for an idle thread
    idle_cond_.Lock();
    while (idle_threads_.empty())
      idle_cond_.Wait();
    // get first idle thread
    if (!idle_threads_.empty()){
      TPoolThr * t = idle_threads_.front();
      idle_threads_.pop_front();
      idle_cond_.Unlock();
      return t;
    }
    idle_cond_.Unlock();
  }
}

//
// append recently finished thread to idle list
//
void TPool::AppendIdle ( TPoolThr * t ){
  // CONSISTENCY CHECK: if given thread is already in list
  idle_cond_.Lock();
  for (std::list< TPoolThr * >::iterator iter = idle_threads_.begin();
         iter != idle_threads_.end(); ++iter ){
    if ((*iter) == t){ return; }
  }
  idle_threads_.push_back( t );
  // wake a blocked thread for job execution
  idle_cond_.Signal();
  idle_cond_.Unlock();
}
//
// to access global thread-pool
//

//parallel run method of each thread in the pool
void TPoolThr::Run(){
  del_mutex_.Lock();
  while (!end_){
    // append thread to idle-list and wait for work
    pool_->AppendIdle( this );
    work_cond_.Lock();
    while ((job_ == NULL ) && !end_ )
      work_cond_.Wait();
    work_cond_.Unlock();
    // look if we really have a job to do and handle it
    if (job_ != NULL) {
      // execute job
      KALDI_LOG << "Thread "<< thread_no_ << " gets job " << job_->JobNo() <<" and is ready for executing it."; 
      job_->Run(data_ptr_);
      KALDI_LOG << "Thread " << thread_no_ << " finished the job.";
      job_->Unlock();
      if (del_job_)
      delete job_;
      // reset data
      work_cond_.Lock();
      job_ = NULL;
      data_ptr_ = NULL;
      work_cond_.Unlock();
    }// if
  }// while
  del_mutex_.Unlock();
  KALDI_LOG << "Thread " << thread_no_ << " quited. ";
}

// set the job for the thread with optional argument
void TPoolThr::SetJob(TPool::Job * j, void * p, const bool del = false ){
  work_cond_.Lock();
  job_ = j;
  data_ptr_ = p;
  del_job_  = del;
  work_cond_.Signal();
  work_cond_.Unlock();  
}

// quit thread (reset data and wake up)
void TPoolThr::Quit(){
  work_cond_.Lock();
  end_ = true;
  job_ = NULL;
  data_ptr_ = NULL;
  work_cond_.Signal();
  work_cond_.Unlock();
}
// init global thread_pool
void Init (const unsigned int  max_p ){
    if (thread_pool != NULL )
        delete thread_pool;
    if ((thread_pool = new TPool( max_p )) == NULL)
        KALDI_LOG << "(init_thread_pool) could not allocate thread pool";
}

// run job
void Run(TPool::Job * job, void * ptr, const bool del ){
  if ( job == NULL )
    return;
  thread_pool->Run(job, ptr, del );
}

// synchronise with specific job
void Sync( TPool::Job * job ){
  thread_pool->Sync(job );
}

// synchronise with all jobs
void SyncAll(){
  thread_pool->SyncAll();
}

// finish thread pool
void Done(){
  delete thread_pool;
}

}// namespace ThreadPool
}// namespace kaldi

