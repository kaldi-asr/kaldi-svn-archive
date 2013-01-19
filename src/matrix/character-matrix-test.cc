#include <iostream>
#include "kaldi-matrix.h"
#include <cstdio>
#include <ctime>
#include <time.h>
#include <sys/timeb.h>
#include "util/kaldi-io.h"

namespace kaldi {

template<class Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.003) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      Real x = A(i, j), y = B(i, j);
      KALDI_ASSERT(std::abs(x - y) <= tol);
      // std::cout << x -y << " ";
    }
    // std::cout << "\n";
  }   
}

template<class T> static void AssertEqual(const CharacterMatrix<T> &A,
                                             const CharacterMatrix<T> &B,
                                             float tol = 0) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j < A.NumCols();j++) {
          T x = A(i, j), y = B(i, j);
	  KALDI_ASSERT(std::abs(x - y) <= tol);
    }
}
//
template <typename Real>
static void Norm(Matrix<Real> &M, Real * NormI, Real* NormII) {
  *NormI = static_cast<Real> (0);
  *NormII = static_cast<Real> (0);
  Real NormMax = static_cast<Real>(0);
  int32 num_rows = M.NumRows(),
        num_cols = M.NumCols();
   for(int32 row = 0; row < num_rows; ++row) {
    *NormI = static_cast<Real>(0) ;
      for(int32 col = 0; col < num_cols; ++col) {
        *NormII += M(row, col) * M(row, col) ;
        *NormI += fabs(M(row, col)) ;
      }
      if(*NormI > NormMax) NormMax = *NormI ;

   }
   *NormII = pow(*NormII, 0.5);
   *NormI = NormMax;
}

//
template <typename Real>
static void  GenerateMatrix4U (Matrix<Real> &M) {
  int32 num_rows = M.NumRows(), 
        num_cols = M.NumCols();
  Real min = static_cast<Real>(0);
  KALDI_ASSERT(num_rows > 0 && num_cols > 0);
  std::vector<Real> v(256);
  for(size_t i = 0; i < v.size(); i++) {
    v[i] = static_cast<Real>(i*1.0/255);
  }
  for(int32 row = 0; row < num_rows; ++row)
    for(int32 col = 0; col < num_cols; ++col) {
      int32  k = rand() % 256;
      M(row, col) = min + v[k];
    }
}
//
template <typename Real>
static void  GenerateMatrix4S (Matrix<Real> &M) {
   int32 num_rows = M.NumRows(), 
         num_cols = M.NumCols();
   KALDI_ASSERT(num_rows > 0 && num_cols > 0);
   Real base = static_cast<Real>(0);
   std::vector<Real> v(256);
   for(int32 i = -128, j=0; i < 128; i++, j++) {
     Real x;
     if(i <= 0) {
       x = static_cast<Real>(i*1.0/128);
     } else {
       x = static_cast<Real>(i*1.0/127);
     }
     v[j] = x;
   }
   for(int32 row = 0; row < num_rows; ++row)
    for(int32 col = 0; col < num_cols; ++col) {
      int32  k = rand() % 256;
      M(row, col) = base + v[k];
    }
}
template <typename Real>
static void  GenerateMatrixI (Matrix<Real> &M) {
   int32 num_rows = M.NumRows(),
         num_cols = M.NumCols();
   KALDI_ASSERT(num_rows > 0 && num_cols > 0);
   srand(time(NULL));
   Real base = static_cast<Real>(0);
   std::vector<Real> v(256);
   for(int32 row = 0; row < num_rows; ++row) {
    for(int32 col = 0; col < num_cols; ++col) {
      if( row == col ) {
         M(row, col) = static_cast<Real>(1) ;
       } else {
    	 M(row, col) = base;
    }
 }
 }
}
// use gdb to display these
template<typename Real>
static void ShowMatrix(const Matrix<Real> &M) {
  std::cout << "\nrow=" << M.NumRows()
            << ", col=" << M.NumCols()
            << "\nmax=" << M.Max() 
            <<", min=" <<M.Min() << "\n";
  for(int32 row = 0; row < M.NumRows(); ++row) {
    for(int32 col = 0; col < M.NumCols(); ++col) {
      std::cout << M(row,col) << " ";
    }
    std::cout << "\n";
  }
}

template<typename Real>
static void ShowMatrix2(std::ostream &os, const Matrix<Real> &M, std::string &note) {
  os << note;
  os << "row=" << M.NumRows()
            << ", col=" << M.NumCols()
            << "\nmax=" << M.Max() 
            <<", min=" <<M.Min() 
            <<", incremental="  << 255.0 / (M.Max() - M.Min()) << "\n";
  for(int32 row = 0; row < M.NumRows(); ++row) {
    for(int32 col = 0; col < M.NumCols(); ++col) {
      os << M(row,col) << " ";
    }
      os << "\n";
  }
}
template<typename OtherReal>
static void ShowMatrixChar(std::ostream &os, const CharacterMatrix<OtherReal> &M, std::string &note) {
  os << note;
  os << "row=" << M.NumRows()
            << ", col=" << M.NumCols() << "\n";
  for(int32 row = 0; row < M.NumRows(); ++row) {
    for(int32 col = 0; col < M.NumCols(); ++col) {
      os << static_cast<int>(M(row,col)) << " ";
    }
      os << "\n";
  }
}

template<typename Real>
static Real NormDiff(std::ostream &os, const Matrix<Real> &M1, 
                     const Matrix<Real> &M2, const std::string  &note) {
  os << note;
  Matrix<Real> diff(M1);
  diff.AddMat(-1.0, M2);

  Real rel_error = diff.FrobeniusNorm() / M1.FrobeniusNorm();
  os << " The relative error is " << rel_error << "\n";
  return rel_error;
}
//
template<typename Real> 
static void TestCopyMat02() {
  int32 num_rows = 2;
  int32 num_cols = 10;
  Matrix<Real> M(num_rows,num_cols);
  Real res = static_cast<Real>(1.0/ 10);
  Real base;
  for(int32 i = 0; i < M.NumRows(); i++) {
     base = 0;
    for(int32 j = 0; j < M.NumCols(); j++) {
      M(i,j) = base + res;
      base += res;
    }
  }
  CharacterMatrix<unsigned char> C;
  C.CopyFromMat(M);
  Matrix<Real> M1;
  C.CopyToMat(&M1);
  for(int32 i = 0; i < M1.NumRows() ; i++) {
    for(int32 j = 0; j < M1.NumCols(); j++) {
      Real x = M1(i,j);
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  AssertEqual(M,M1);
}

template<typename Real>
static void TestAddMatMatError(int32 MatNum ) {
  Real error_avg = static_cast<Real> (0);
  Real error_std = static_cast<Real> (0);
  std::ostringstream os;
  os << "temp_err_sum_" << MatNum;
  Output ko(os.str(), false, false);
 
  srand(time(NULL));
  ko.Stream() << " Number Of Matrices : " << MatNum << "\n";
  std::vector<Real> v(MatNum);
  for(int32 i = 0; i< MatNum; ++i) {
    // srand(time(NULL));
    int32  row = 100 + rand() % 10;
    int32  col = 100 + rand() % 10;
    Matrix<Real> M1(row, col);
    M1.SetRandn();
    // GenerateMatrix4U(M1);
    ko.Stream() << "\n=================" << "round " << i+1 << "==========================\n";
    std::string note("---------M1----------\n");
    //ShowMatrix2<Real>(ko.Stream(), M1, note); 

    int32 row2 = 100 + rand() % 10;
    Matrix<Real> M2(row2,col);
    M2.SetRandn();
    // GenerateMatrix4S(M2);
    std::string note1("--------M2-------------\n");
    //ShowMatrix2<Real>(ko.Stream(), M2, note1);
    Matrix<Real> M(row, row2);
    M.AddMatMat(1.0, M1, kNoTrans, M2, kTrans, 0);
   
    CharacterMatrix<unsigned char> Mc1;
    Mc1.CopyFromMat(M1);
    // let us take a look what has happened 
    // when we copy back from char to float matrix
    Matrix<Real> Mc2r1;
    Mc1.CopyToMat(&Mc2r1);
    std::string note3("---------Mc2r1 (from Mc1)----------\n");
    //ShowMatrix2<Real>(ko.Stream(), Mc2r1, note3);
    std::string note4("--------M1 vs Mc2r1 -------------\n");
    NormDiff<Real>(ko.Stream(), M1, Mc2r1, note4);

    CharacterMatrix<signed char> Mc2;
    Mc2.CopyFromMat(M2);
   
    Matrix<Real>Mc2r2;
    Mc2.CopyToMat(&Mc2r2);
    std::string note5("---------Mc2r2 (from Mc2)-----------\n");
    //ShowMatrix2<Real>(ko.Stream(), Mc2r2, note5);
    std::string note7("--------M2 vs Mc2r2 -------------\n");
    NormDiff<Real>(ko.Stream(), M2, Mc2r2, note7);

    Matrix<Real> Mc(row,row2);
    Mc.AddMatMat(1.0, Mc1, kNoTrans, Mc2, kTrans, 0);
    std::string note6("-----------M vs Mc --------\n");
    Real rel_error = NormDiff<Real>(ko.Stream(), M, Mc, note6);
    error_avg += rel_error;
     
    // test  Sse4DotProduct function, this should be separately tested
    Matrix<Real> Mc_naked2(row,row2);
    Mc_naked2.AddMatMat2(1.0, Mc1, kNoTrans, Mc2, kTrans, 0);
    std::string note8("---------Mc vs Mc_naked(for Sse4DotProduct)----\n");
    NormDiff<Real>(ko.Stream(), Mc, Mc_naked2, note8);

    sleep(1);
  }
  error_avg = error_avg/MatNum;
  for(int32 i = 0; i< MatNum; ++i) {
     error_std += pow((v[i]-error_avg), 2);
  }
  error_std = pow( error_std/MatNum, 0.5);
  ko.Stream() << " Average Error is : " 
     << error_avg << " Standard Deviation : "
     << error_std << std::endl; 
  ko.Close();
  std::cout << " Average Error is : " 
     << error_avg << " Standard Deviation : "
     << error_std << std::endl; 
}
//
template<typename Real>
static void TestAddMatMatTime (int32 numTest) {
  std::ostringstream os;
  os << "temp_time_sum_" << numTest;
  Output ko(os.str(), false,false);
  int num_threads = 8;
  struct timeb tstruct;
  int tstart = 0, tend = 0;
  double tot_ft1 = 0, 
         tot_ft2 = 0,
 	 tot_ft3 = 0;
  for(int32 i =0; i < numTest; i++) {
    ko.Stream() << "\nround " << i+1 << "\n";
    int32  row = 2000;// + rand() % 5;
    int32  col = 2000;// + rand() % 7;

    Matrix<Real> M1(row, col);
    GenerateMatrix4U(M1);
    int32 row2 = 2000;// + rand() % 4;
    Matrix<Real> M2(row2,col);
    GenerateMatrix4S(M2);
    Matrix<Real> M(row,row2);
    //clock_t start = std::clock();
    //time_t start1, end1;
    //time(&start1);
    ftime( &tstruct );
    tstart = tstruct.time * 1000 + tstruct.millitm;
    M.AddMatMat(1.0, M1, kNoTrans, M2, kTrans, 0);
    ftime( &tstruct );
    tend = tstruct.time * 1000 + tstruct.millitm; 
    //time(&end1);
    tot_ft1 = tend - tstart;
    //tot_ft1 += difftime (end1,start1);
    //tot_ft1 += (std::clock() - start) / (double)CLOCKS_PER_SEC;

    ko.Stream() << "\nMax=" << M.Max()
              << ", Min=" << M.Min() << "\n";

    CharacterMatrix<unsigned char> Mc1;
    Mc1.CopyFromMat(M1);
    CharacterMatrix<signed char> Mc2;
    Mc2.CopyFromMat(M2);
    Matrix<Real> Mc(row,row2);
    //start = std::clock();   
    //time_t start2, end2;
    //time(&start2);
    ftime( &tstruct );
    tstart = tstruct.time * 1000 + tstruct.millitm;
    Mc.AddMatMat(1.0, Mc1, kNoTrans, Mc2, kTrans, 0);
    ftime( &tstruct );
    tend = tstruct.time * 1000 + tstruct.millitm; 
    //time(&end2);
    tot_ft2 = tend - tstart;
    //tot_ft2 += difftime (end2,start2);
    //tot_ft2 += (std::clock() - start) / (double)CLOCKS_PER_SEC;
    ftime( &tstruct );
    tstart = tstruct.time * 1000 + tstruct.millitm;
    Mc.AddMatMatPthread(1.0, Mc1, kNoTrans, Mc2, kTrans, 0, num_threads);
    ftime( &tstruct );
    tend = tstruct.time * 1000 + tstruct.millitm; 
    tot_ft3 = tend - tstart;

    ko.Stream() << "\ncMax=" << Mc.Max()
              << ", cMin=" << Mc.Min() << "\n";

    sleep(1);
  }
  ko.Stream() << "\nfloat_AddMatMat=" << tot_ft1 
            << ", char_AddMatMat=" << tot_ft2 
            << ",rate=" << tot_ft2/tot_ft1 << "\n";
  std::cout << "Time consumed in milliseconds: float_AddMatMat = " << tot_ft1 << 
  ", char_AddMatMat (single threaded) = " << tot_ft2 << ", char_AddMatMat with " <<
   num_threads << " threads = " << tot_ft3 << ", rate = " << tot_ft3/tot_ft1 << std::endl;
  ko.Close();
}

template<typename Real>
static void TestError2(int32 MatNum ) {
  Real error_avg = static_cast<Real> (0);
  Real error_std = static_cast<Real> (0);
  std::ostringstream os;
  os << "temp_err_sum_" << MatNum;
  Output ko(os.str(), false, false);

  srand(time(NULL));
  ko.Stream() << " Number Of Matrices : " << MatNum << "\n";
  std::vector<Real> v(MatNum);
  for(int32 i = 0; i< MatNum; ++i) {
  ko.Stream() << " Matrix number = "<<i<<"\n";
  int32 row = 4;
  int32 col = 8;
  int32 row2 = 16;
  Matrix<Real> M1(row, col);
  //M1.SetRandn();
  
  M1(0,0) = 1;M1(0,1) = 2;M1(1,1) = 1;M1(1,0) = 2; 
  M1(2,2) = 1;
  //M1(2,3) = 2;M1(3,3) = 1;
  M1(3,2) = 2; 

  std::string note1("---------M1----------\n");
  ShowMatrix2<Real>(ko.Stream(), M1, note1);
  Matrix<Real> M2(row2, col);
  //M2.SetRandn();

  M2(0,0) = 1;
  M2(1,1) = 1;
  M2(2,2) = 1;
  M2(3,3) = 1;
  M2(3,4) = 1;
   
  std::string note2("---------M2----------\n");
  ShowMatrix2<Real>(ko.Stream(), M2, note2);
  Matrix<Real> M(row, row2);
  M.AddMatMat(1.0, M1, kNoTrans, M2, kTrans, 0);
  std::string note3("---------M----------\n");
  ShowMatrix2<Real>(ko.Stream(), M, note3);
  CharacterMatrix<unsigned char> Mc1;
  Mc1.CopyFromMat(M1);
  std::string note4("---------Mc1----------\n");
  ShowMatrixChar<unsigned char>(ko.Stream(), Mc1, note4);
  CharacterMatrix<signed char> Mc2;
  Mc2.CopyFromMat(M2);
  std::string note5("---------Mc2----------\n");
  ShowMatrixChar<signed char>(ko.Stream(), Mc2, note5);
  Matrix<Real> Mc(row, row2);
  Mc.AddMatMat(1.0, Mc1, kNoTrans, Mc2, kTrans, 0);
  std::string note6("---------Mc2----------\n");
  ShowMatrix2<Real>(ko.Stream(), Mc, note6);
  //ko.Stream()  << " Mc : " << ShowMatrix2(Mc) <<"\n" ;
  std::cout<<"float_AddMatMat = "<<M(0,0)<<"char_AddMatMat = "<<Mc(0,0)<<std::endl;
  
  }
 }
 template<typename Real>
 static void TestSse4DotProduct(int MatNum) {
 int32 row  = 4;
 int32 col = 8;
 int32 row2 = 15;
 std::ostringstream os;
  os << "Sse4DotProduct_test_" << MatNum;
  Output ko(os.str(), false, false);
 for (int32 i = 0; i < MatNum; ++i) {
 ko.Stream() << " Matrix Number = " << i <<"\n" ;
 Matrix<Real> Mr1(row, col);
 Mr1.SetRandn();
 CharacterMatrix<unsigned char> M1;
 M1.CopyFromMat(Mr1);
 std::string note4("---------Mc1----------\n");
 //ShowMatrixChar<unsigned char>(ko.Stream(), Mc1, note4);
 Matrix<Real> Mr2(row2, col);
 Mr2.SetRandn();
 CharacterMatrix<signed char> M2;
 M2.CopyFromMat(Mr2);
 std::string note5("---------Mc2----------\n");
 //ShowMatrixChar<signed char>(ko.Stream(), Mc2, note5);
 Matrix<Real> M4(row, row2);
 Matrix<Real> M3(row, row2);
 Matrix<Real> Md(row, row2);
 Matrix<Real> Md1(row, row2);
 
 /*
 int x1 ;
 x1 = Sse4DotProduct(reinterpret_cast<unsigned char*>(Mc1.Data())  ,reinterpret_cast<signed char*>(Mc2.Data()) , col);
 ko.Stream() << " Sse4DotProduct test, x1 = "<< x1 << "\n" ;
 x1 = DotProduct(reinterpret_cast<unsigned char*>(Mc1.Data())  ,reinterpret_cast<signed char*>(Mc2.Data()) , col);
 ko.Stream() << " DotProduct test, x1 = "<< x1 << "\n" ;
 */
  float beta = 0; float alpha = 1;

  // pre-calculate some constant
  float mul_inc = M1.Increment() * M2.Increment(),
  low_t2 = static_cast<float>(std::numeric_limits<signed char>::min()),
  coef1 = M2.Min() / M1.Increment() - low_t2 /mul_inc,
  coef2 = M1.Min() / M2.Increment() ,
  gconst = M1.Min() * M2.Min()  - M1.Min() * low_t2 / M2.Increment();
  CharacterMatrix<signed char> Mt;
  Mt.Resize(1, M1.NumCols());
  for(int32 col = 0; col < M1.NumCols(); ++col) {
    *(Mt.Data() + col) = static_cast<signed char>(1);
  }

  int x3[M2.NumRows()];
  for (MatrixIndexT col = 0; col < M2.NumRows(); ++col){
    x3[col] = Sse4DotProduct(reinterpret_cast<unsigned char*>(Mt.Data()), M2.Data() + col * M2.Stride(), M1.NumCols());
    //x3[col] = DotProduct(reinterpret_cast<unsigned char*>(Mt.Data()), M2.Data() + col * M2.Stride(), M1.NumCols());
    //x3[col] = Sse4SumArray(M2.Data() + col * M2.Stride(), M1.NumCols());
  }

  for(MatrixIndexT row = 0; row < M1.NumRows(); ++ row) {
    int x2 = Sse4DotProduct(M1.Data() + row *M1.Stride(), Mt.Data(), M1.NumCols());
    //int x2 = DotProduct (M1.Data() + row *M1.Stride(), Mt.Data(), M1.NumCols());
    //int x2 = Sse4SumArray(M1.Data() + row *M1.Stride(), M1.NumCols());
    MatrixIndexT col = 0;
    
    for( col = 0; col+3 < M2.NumRows(); col += 4) {
        int x1[4];
        x1[0] = 0;
        x1[1] = 0;
        x1[2] = 0;
        x1[3] = 0;
        Sse4DotProduct4fold1X4(M1.Data() + row * M1.Stride(),
                          M2.Data() + col * M2.Stride(), M2.Data() + (col + 1) * M2.Stride(),
                          M2.Data() + (col + 2) * M2.Stride(), M2.Data() + (col + 3) * M2.Stride(), x1,  M1.NumCols());
        
        
        float *this_data  = (M3.Data() + row * M3.Stride() + col);
        
        *this_data = static_cast<float>( beta * (*this_data) +
                                        alpha * (static_cast<float>(x1[0]) / mul_inc +
                                                 coef1 * x2 + coef2 * x3[col] + gconst * M1.NumCols() ));
        *(this_data + 1) = static_cast<float>( beta * (*(this_data + 1)) +
                                              alpha * (static_cast<float>(x1[1]) / mul_inc +
                                                       coef1 * x2 + coef2 * x3[col+1] + gconst * M1.NumCols() ));
        *(this_data + 2) = static_cast<float>( beta * (*(this_data + 2)) +
                                              alpha * (static_cast<float>(x1[2]) / mul_inc +
                                                       coef1 * x2 + coef2 * x3[col+2] + gconst * M1.NumCols() ));
        *(this_data + 3) = static_cast<float>( beta * (*(this_data + 3)) +
                                              alpha * (static_cast<float>(x1[3]) / mul_inc +
                                                       coef1 * x2 + coef2 * x3[col+3] + gconst * M1.NumCols() ));
    }
    
    
      for(col = col; col < M2.NumRows(); ++col) {
        int x1 = Sse4DotProduct(M1.Data() + row * M1.Stride(),
                                M2.Data() + col * M2.Stride(), M1.NumCols());
        //int x1 = DotProduct(M1.Data() + row * M1.Stride(),
        //                         M2.Data() + col * M2.Stride(), M1.NumCols());
        
        
        float *this_data  = (M3.Data() + row * M3.Stride() + col);
        *this_data = static_cast<float>( beta * (*this_data) +
                                        alpha * (static_cast<float>(x1) / mul_inc +
                                                 coef1 * x2 + coef2 * x3[col] + gconst * M1.NumCols() ));
      }
    }

    for(MatrixIndexT row = 0; row < M1.NumRows(); ++ row) {
      int x2 = Sse4DotProduct(M1.Data() + row *M1.Stride(), Mt.Data(), M1.NumCols());
      //int x2 = DotProduct (M1.Data() + row *M1.Stride(), Mt.Data(), M1.NumCols());
      //int x2 = Sse4SumArray(M1.Data() + row *M1.Stride(), M1.NumCols());
      MatrixIndexT col = 0;

      for(col = col; col < M2.NumRows(); ++col) {
        int x1 = Sse4DotProduct(M1.Data() + row * M1.Stride(),
                                  M2.Data() + col * M2.Stride(), M1.NumCols());
        //int x1 = DotProduct(M1.Data() + row * M1.Stride(),
        //                         M2.Data() + col * M2.Stride(), M1.NumCols());

        float *this_data  = (M4.Data() + row * M4.Stride() + col);
        *this_data = static_cast<float>( beta * (*this_data) +
                                        alpha * (static_cast<float>(x1) / mul_inc +
                                                 coef1 * x2 + coef2 * x3[col] + gconst * M1.NumCols() ));
      }
    } 
    Matrix<Real> Mc_naked2(row,row2);
    Mc_naked2.AddMatMat2(1.0, M1, kNoTrans, M2, kTrans, 0);
    
    for(MatrixIndexT row = 0; row < M3.NumRows(); ++row){
      for(MatrixIndexT col = 0; col < M3.NumCols(); ++col){
        Md(row,col) = M3(row,col)-M4(row,col);
        Md1(row,col) = M4(row,col)-Mc_naked2(row,col);
      }
    }
    
  std::string note1("---------M3----------\n");
  ShowMatrix2<Real>(ko.Stream(), M3, note1);
  std::string note2("---------M4----------\n");
  ShowMatrix2<Real>(ko.Stream(), M4, note2);
  std::string note3("---------Md----------\n");
  ShowMatrix2<Real>(ko.Stream(), Md, note3);
  std::string note8("---------Md1----\n");
  ShowMatrix2<Real>(ko.Stream(), Md1, note8);
  //std::string note6("---------Mc2----------\n");
  //ShowMatrix2<Real>(ko.Stream(), M4, note4);
  }
  }
static void MatMatNaive(Matrix<float> &m, CharacterMatrix<unsigned char> &m1,
                CharacterMatrix<signed char> &m2) {

  for(int32 row = 0; row < m1.NumRows(); ++ row) {
    unsigned char * mp1 = m1.Data() + row * m1.Stride();
    for(int32 col = 0; col < m2.NumRows(); ++ col) {
      signed char *mp2 = m2.Data() + col * m2.Stride();
      float &x = m(row,col);
      x = static_cast<float>(DotProduct(mp1,mp2, m1.NumCols()));
    }
  }
}
static void MatMatBlockingTest(const int32 numTest) {

  float naive_total = 0, block_total = 0;
  for (int32 i = 0; i < numTest; ++i) {
    int32  row = 2000 + rand() % 5;
    int32  col = 2000 + rand() % 7;

    Matrix<float> mf1(row, col);
    GenerateMatrix4U(mf1);
    int32 row2 = 400 + rand() % 4;
    Matrix<float> mf2(row2,col);
    GenerateMatrix4S(mf2);

    CharacterMatrix<unsigned char> mc1, mc11;
    int32 blk_num_rows = 125 + rand() % 40;
    int32 blk_num_cols = 165 + rand() % 40; 
    mc1.BlockResize(blk_num_rows, blk_num_cols);
    mc1.CopyFromMat(mf1);
    mc1.CheckMatrix();
    
    mc11.CopyFromMat(mf1);
    AssertEqual(mc1,mc11); 

    CharacterMatrix<signed char> mc2, mc21;
    blk_num_rows = 40 + rand() % 10;    
    mc2.BlockResize(blk_num_rows, blk_num_cols);
    mc2.CopyFromMat(mf2);
    mc2.CheckMatrix();
   
    mc21.CopyFromMat(mf2);
    AssertEqual(mc2,mc21);   

    Matrix<float> mfx1(row, row2);
    clock_t start = std::clock();
    MatMatNaive(mfx1, mc1, mc2);
    naive_total += (std::clock() - start) / (double)CLOCKS_PER_SEC;
   
    Matrix<float> mfx2(row, row2);
    start = std::clock();
    mfx2.AddMatMat2(1.0f, mc1, kNoTrans, mc2, kTrans, 0, true);
    block_total += (std::clock() - start) / (double)CLOCKS_PER_SEC;

    AssertEqual(mfx1, mfx2, 0);
  } // end i
  std::cout << "naive=" << naive_total <<", block=" << block_total <<"\n";
}

} // kaldi namespace

int main() {
  // kaldi::TestAddMatMatError<float>(1);
  // kaldi::TestAddMatMatTime<float>(1);
  // kaldi::TestSse4DotProduct<float>(1); 
  // kaldi::TestError2<float>(3);
  kaldi::MatMatBlockingTest(3);
  //kaldi::TestAddVecMat<float>();
  // kaldi::TestAddMatMatParallel<float>();
  KALDI_LOG << "character-matrix-test succeeded.\n";
  return 0;
}
