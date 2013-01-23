#include <cstdlib>
#include "threadpool.h"

namespace kaldi {
class TMyJob : public ThreadPool::TPool::Job{
 protected:
  int arg_;
 public:
  TMyJob(int i,int s):ThreadPool::TPool::Job(i),arg_(s) {}
  virtual void Run (void *){
    KALDI_LOG<<"Excuting the "<< arg_ <<" th job.";
  }
};
}// namespace kaldi

int main (){
  int num_jobs = 10;
  using namespace kaldi;
  ThreadPool::Init(5);
  
  TMyJob ** jobs = new TMyJob* [num_jobs];
  for (int i = 0; i < num_jobs; i++) jobs[i] = new TMyJob(i, i);
  for (int i = 0; i < num_jobs; i++) ThreadPool::Run(jobs[i]);
  KALDI_LOG<<"All jobs are finished";
  for (int i = 0; i < num_jobs; i++) ThreadPool::Sync(jobs[i]);
  KALDI_LOG<<"Synconization of all threads finished.";
  ThreadPool::Done();
  for (int i = 0; i < num_jobs; i++) delete jobs[i];
  delete[] jobs;
  KALDI_LOG <<"Task finshed.";
}
