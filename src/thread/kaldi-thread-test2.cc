// util/kaldi-thread-test3.cc

// Copyright 2012  Johns Hopkins University (Author: Pegah Ghahremani)


#include "base/kaldi-common.h"
#include "thread/kaldi-thread.h"
#include "thread/kaldi-semaphore.h"
#include "matrix/kaldi-matrix.h"
namespace kaldi {

template<typename Real>
class MyThreadClass2 {  // Doing Matrix multiplication.
 public:
  MyThreadClass2(CharacterMatrix<unsigned char> &M1, CharacterMatrix<signed char> &M2,
                                          Matrix<Real> &M3,  
                                          Matrix<Real> *i): row_num_(M1.NumRows()),
                                          col_num_(M2.NumRows()),iptr_(i)
                                               { 
                                                // Matrix<Real> Mc(row_num_,col_num_);
                                                // Mc.SetRandn();
                                                // private_counter_ = &Mc;
                                                 private_counter_.Resize(row_num_, col_num_);
                                                 M1_ = &M1;
                                                (*M1_).CopyFromMatrix(M1);
                                                 M2_ = &M2;
                                                (*M2_).CopyFromMatrix(M2);
                                              // (*test1).Resize(10,10);
                                               // (*M2_).Resize(10,10);
                                               std::cout << " M1_(1,1) = " << static_cast<float>((*M1_)(1,1))
                                               << " M1(1,1) = " << static_cast<float>((M1)(1,1))<< std::endl;
                                              // M1_.NumCols() <<M2_.NumRows() << M2_.NumCols() << std::endl;
                                               }
  // Use default copy constructor and assignment operators.
  void operator() () {
    int32 block_size = (row_num_ ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(row_num_, start + block_size);
    std::cout << " M1 Outside = " << (*M1_).NumRows() << 
    " " << (*M1_).NumCols() << " M1_(1,1) = " << static_cast<float>((*M1_)(1,1)) << std::endl;
    std::cout << "start and end = " << start <<" , " << end << std::endl;
    private_counter_.AddVecMat(1, M1_, kNoTrans, M2_, kTrans, 0 , start, end);
   //  private_counter_.AddMatMat(1,(*M1_),kNoTrans,(*M2_),kTrans, 0);
    std::cout << " private_counter_ = " << (private_counter_).NumRows() << std::endl;
   }
  ~MyThreadClass2() {
    
   (*iptr_).AddMat(1,(private_counter_),kNoTrans);
  // std::cout << "iptr_ = " << (*iptr_).NumRows() << ", " << (*iptr_).NumCols() << std::endl;
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
  Matrix<Real> *test1;// = new  Matrix<Real>[1];
  CharacterMatrix<unsigned char> *M1_;// = new CharacterMatrix<unsigned char>;
  CharacterMatrix<signed char> *M2_ ;//= new CharacterMatrix<signed char>;
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
 int32 row_num = 1000;
 int32 col = 1000 ;
 int32 row2 = 1000;
 Matrix<Real> Mr1(row_num, col);
 Mr1.SetRandn();
 CharacterMatrix<unsigned char> M1;
 M1.CopyFromMat(Mr1);

 Matrix<Real> Mr2(row2, col);
 Mr2.SetRandn();
 CharacterMatrix<signed char> M2;
 M2.CopyFromMat(Mr2);
// Matrix<Real> tot_true(row_num, row2);
// Matrix<Real> tot_2(row_num, row2);
// tot_2.AddMatMat(1.0,M1,kNoTrans,M2,kTrans,0);
// tot_true.AddMatMat(1.0,Mr1,kNoTrans,Mr2,kTrans,0);
 Matrix<Real> tot(row_num, row2);
 MyThreadClass2<Real> c(M1,M2,Mr2, &tot);
/*
pthread_attr_t pthread_attr;
 pthread_attr_init(&pthread_attr);
 pthread_t *threads1 = new pthread_t[num_threads];
 MyThreadClass2<Real> *m ; //new MyThreadClass2<Real>[num_threads];
 for(int32 thread = 0; thread < 1; thread++) {
 c.thread_id_ = thread;
 c.num_threads_ = num_threads;
 int32 ret;

 if((ret=pthread_create(&(threads1[thread]),&pthread_attr,MyThreadClass2<Real>::run,&c))) {
   const char *c = strerror(ret);
   if (c == NULL) { c = "[NULL]"; }
   KALDI_ERR << "Error creating thread, errno was: " << c;
 }
 if(pthread_join(threads1[thread],NULL))
   KALDI_ERR << "Error rejoining thread.";
 }
 delete [] threads1;
*/ 
 RunMultiThreaded (c);
 //std::cout << "tot(0,8) = " << tot(0,8) << "true value = " << tot_true(0,8) << std::endl;
 } 


// End of Test;
}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  //TestThreadsSimple();
   TestThreads2<float>(100);
}

