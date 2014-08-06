// cudamatrix/kaldi-tensor-test.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2014  Pegah Ghahremani
//           2014  Vijayaditya Peddinti
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "matrix/kaldi-tensor.h"
#include "matrix/kaldi-tensor-internals.h"
#include "cudamatrix/cu-tensor.h"
#include "cudamatrix/cu-matrix-lib.h"

namespace kaldi {

template<typename Real>
void TestScale() {
  KALDI_LOG << "=== TestScale() ===\n";
  typedef std::pair<int32, int32> DimsStrides;
  { 
    // Scales elements of 2D-Cutensor and 2D-tensor and checks if they are equal
    KALDI_LOG << "Scale 2D-Cutensor ";
    for (int t = 0; t < 10; t++) {
      int n = 20 + rand() % 10, m = 40 + rand() % 10;
      Real scaling_coeff = rand();

      std::vector<DimsStrides> dims_strides;
      dims_strides.push_back(DimsStrides(n, 1));
      dims_strides.push_back(DimsStrides(m, n));
      Matrix<Real> m1(1, n * m);
      m1.SetRandn();
      
      CuMatrix<Real> cu_m1(m1); 
      CuTensor<Real> ct1(dims_strides, cu_m1);
      Tensor<Real> t1(dims_strides, m1);
      ct1.Scale(scaling_coeff);
      t1.Scale(scaling_coeff);
       
      // Compares the individual elements of the array with corresponding tensor
      KALDI_ASSERT(ct1.ApproxEqual(t1));

    }
  }
  {
    // Scales elements of 3D-CuTensor and 3D-Tensor and checks if they are equal
    KALDI_LOG << " Scale 3D-tensor ";
    for (int t = 0; t < 10; t++) {
      int n = 20 + rand() % 10, m = 40 + rand() % 10,
        k = 10 + rand() % 5;
      float scaling_coeff = rand();
      std::vector<DimsStrides> dims_strides;
      dims_strides.push_back(DimsStrides(n, 1));
      dims_strides.push_back(DimsStrides(m, n));
      dims_strides.push_back(DimsStrides(k, n * m));
     
      KALDI_LOG<<n<<" "<<m<<" "<<k;
      CuMatrix<Real> m1(1, n * m * k);
      Matrix<Real> m2(1, n * m * k); 
      m2.SetRandn();
      m1.CopyFromMat(m2);
      CuTensor<Real> ct1(dims_strides, m1);
      Tensor<Real> t1(dims_strides, m2);
      t1.Scale(scaling_coeff);
      ct1.Scale(scaling_coeff);

      KALDI_ASSERT(ct1.ApproxEqual(t1));
    }
  }
  KALDI_LOG<<__func__<<" passed! ";
}
 // Not available as GetTensor() is a protected method.
/*template<class Real>
void TestGetTensor() {
 typedef std::pair<int32, int32> DimsStrides;
 std::vector<DimsStrides> dims_strides;
 int d1 = 10, d2 = 3, d3 = 5;
 dims_strides.push_back(DimsStrides(d1, 1));
 dims_strides.push_back(DimsStrides(d2, d1));
 dims_strides.push_back(DimsStrides(d3, d1 * d2));  
 
 Matrix<Real> m(1, d1 * d2 * d3);
 m.SetRandn();

 CuTensor<Real> ct(dims_strides, CuMatrix<Real>(m));
 std::vector<int> indices;
 indices.push_back(1);
 indices.push_back(2);
 indices.push_back(3);
 // Checks if all the elements in the tensor are accessible  
 for (int i=0; i < d1; i++) {
   for (int j=0; j < d2; j++) {
     for (int k=0; k < d3; k++) {
       indices[0]=i; indices[1]=j; indices[2]=k;
        ct(indices)=static_cast<Real>(i*j*k);
     }
   }
 }
 Tensor<Real> t = ct.GetTensor();
 for (int i=0; i < d1; i++) {
   for (int j=0; j < d2; j++) {
     for (int k=0; k < d3; k++) {
       indices[0]=i; indices[1]=j; indices[2]=k;
       KALDI_ASSERT(t(indices)==i*j*k);
     }
   }
 }
 KALDI_LOG<< " Tensor access passed.";
}*/

template<class Real>
void TestCopyFromTensor() {
 typedef std::pair<int32, int32> DimsStrides;
 CuMatrix<Real> m1(1,150), m2(1,150);
 Matrix<Real> m3(1,150);
 m3.SetRandn();
 m1.CopyFromMat(m3);

 std::vector<DimsStrides> dims_strides;
 dims_strides.push_back(DimsStrides(10,1));
 dims_strides.push_back(DimsStrides(3,10));
 dims_strides.push_back(DimsStrides(5,30));
 
 CuTensor<Real> ct(dims_strides, m1);
 CuTensor<Real> ct2(dims_strides, m2);

 ct2.CopyFromTensor(ct);
 
 KALDI_ASSERT(ct2.ApproxEqual(ct));

 KALDI_LOG<<__func__<<" test passed!";
}

template<typename Real>
void TestAddTensor() {
  KALDI_LOG << "=== TestAddTensor() ===\n";  
  typedef std::pair<int32, int32> DimsStrides;
  {
    // Sums elements of two cu_tensors and compares the sum with the sum of
    // corresponding tensors
    for (int t = 0; t < 10; t++) {
      int32 n = 20 + rand() % 10, m = 40 + rand() % 20;

      Real alpha = rand();

      std::vector<DimsStrides> A_dims_strides, B_dims_strides;
      A_dims_strides.push_back(DimsStrides(n,1));
      A_dims_strides.push_back(DimsStrides(m,n));
      Matrix<Real> A_mat(1, n * m), B_mat(1, n * m);
      A_mat.SetRandn();
      B_mat.SetRandn();
      CuMatrix<Real> A_mat2(A_mat), B_mat2(B_mat);

      CuTensor<Real> A_cutensor(A_dims_strides, A_mat2);
      CuTensor<Real> B_cutensor(A_dims_strides, B_mat2);
      A_cutensor.AddTensor(alpha, B_cutensor);
  
      Tensor<Real> A_tensor(A_dims_strides, A_mat);
      Tensor<Real> B_tensor(A_dims_strides, B_mat);
      A_tensor.AddTensor(alpha, B_tensor);
      
      KALDI_ASSERT(A_cutensor.ApproxEqual(A_tensor));
    }
  }
}

template<typename Real>
void TestAddTensorTensor() {
  KALDI_LOG << "=== TestAddTensorTensor() ===\n";
  typedef std::pair<int32, int32> DimsStrides;
    for (int t = 0; t < 20; t++) { 
      // matrix multiplication using tensor product
      int n = 50 + rand() % 20, m = 50 + rand() % 20, k = 100 + rand() % 40;
      std::vector<DimsStrides> A_dims_strides, B_dims_strides,
        C_dims_strides;
      A_dims_strides.push_back(DimsStrides(n,1));
      A_dims_strides.push_back(DimsStrides(k,n));
      A_dims_strides.push_back(DimsStrides(1,n * k));
      B_dims_strides.push_back(DimsStrides(1,1));
      B_dims_strides.push_back(DimsStrides(k,1));
      B_dims_strides.push_back(DimsStrides(m,k));
      C_dims_strides.push_back(DimsStrides(n,1));
      C_dims_strides.push_back(DimsStrides(1,n));
      C_dims_strides.push_back(DimsStrides(m,n));

      Matrix<Real> A_mat(1, n * k), B_mat(1, m * k),
        C_mat(1, n * m);
      A_mat.SetRandn(); B_mat.SetRandn();
      CuMatrix<Real> A_mat2(A_mat), B_mat2(B_mat), C_mat2(C_mat);

      CuTensor<Real> A_cutensor(A_dims_strides, A_mat2),
        B_cutensor(B_dims_strides, B_mat2),
        C_cutensor(C_dims_strides, C_mat2);

      Tensor<Real> A_tensor(A_dims_strides, A_mat),
        B_tensor(B_dims_strides, B_mat),
        C_tensor(C_dims_strides, C_mat);

      C_cutensor.AddTensorTensor(1.0, A_cutensor, B_cutensor, 0.0);
      C_tensor.AddTensorTensor(1.0, A_tensor, B_tensor, 0.0);

      KALDI_ASSERT( C_cutensor.ApproxEqual(C_tensor));
    }
}

template<typename Real> 
void TestConvTensorTensor() {
  KALDI_LOG << "=== TestConvTensorTensor() ===\n";
  typedef std::pair<int32, int32> DimsStrides;
  {
    // tests 2D convolution of 2D cutensors 
    // Checks if ConvTensorTensor(t1, t2) = ConvTensorTensor(t2, t1)
    // Checks if the output is similar to those of Tensors
    // d3 + 1 = d1 + d2 
    for (int32 t = 0; t < 10; t++) {
      int n1 = 20 + rand() % 10, m1 = 30 + rand() % 10, 
        n2 = 10+ rand() % 5 , m2 = 15 + rand() % 5,
        n3 = n1 + n2 - 1, m3 = m1 + m2 - 1;
      std::vector<DimsStrides> A_dims_strides, B_dims_strides, C_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1,1));
      A_dims_strides.push_back(DimsStrides(m1,n1));

      B_dims_strides.push_back(DimsStrides(n2,1));
      B_dims_strides.push_back(DimsStrides(m2,n2));

      C_dims_strides.push_back(DimsStrides(n3,1));
      C_dims_strides.push_back(DimsStrides(m3,n3));

      Matrix<Real> A_mat(1, n1 * m1), B_mat(1, n2 * m2),
                   C_mat(1, n3 * m3);
      A_mat.SetRandn();
      B_mat.SetRandn();
      CuMatrix<Real> A_mat2(A_mat), B_mat2(B_mat), 
        C_mat2(C_mat), C_mat3(C_mat);
      CuTensor<Real> A_cutensor(A_dims_strides, A_mat2),
        B_cutensor(B_dims_strides, B_mat2),
        C_cutensor(C_dims_strides, C_mat2),
        C_cutensor2(C_dims_strides, C_mat3);
      
      // Checks if the two convolution outputs are equal
      C_cutensor.ConvTensorTensor(1.0, A_cutensor, B_cutensor);
      C_cutensor2.ConvTensorTensor(1.0, B_cutensor, A_cutensor);
      C_cutensor.ApproxEqual(C_cutensor2); 

      // Compares the output with a tensor's ConvTensorTensor operation
      Tensor<Real> A_tensor(A_dims_strides, A_mat),
        B_tensor(B_dims_strides, B_mat),
        C_tensor(C_dims_strides, C_mat);
      C_tensor.ConvTensorTensor(1.0, A_tensor, B_tensor); 

      C_cutensor.ApproxEqual(C_tensor);
      
    }
  }
}
template<typename Real>
static void TestOperator() {
  typedef std::pair<int32, int32> DimsStrides; 
  {
    for (int t = 0; t < 1; t++) {
      int32 n1 = 4 + rand() % 10, m1 = n1 + 5;
      std::vector<DimsStrides> A_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1,1));
      A_dims_strides.push_back(DimsStrides(m1,n1));    
      CuMatrix<Real> A_mat(1, n1 * m1);
      A_mat.SetRandn();
      CuTensor<Real> A_tensor(A_dims_strides, A_mat);
      Real tensor_sum = 0;
      std::vector<int> index(2,0);
      for (int i = 0; i < n1; i++) {
        index[0] = i;
        for (int j = 0; j < m1; j++) {
          index[1] = j;
          tensor_sum += A_tensor(index);
        }
      }
      KALDI_LOG << " tensor .vs. matrix sum " << tensor_sum << " " << A_mat.Sum();
      AssertEqual(tensor_sum, A_mat.Sum());
    }
  }
}
template<class Real>
void CuTensorUnitTest() {
  //TestFlatten<Real>();
  //TestGetTensor<Real>();
  TestCopyFromTensor<Real>();
  TestScale<Real>();
  TestAddTensor<Real>();
  TestAddTensorTensor<Real>();
  TestConvTensorTensor<Real>();
  TestOperator<Real>();
}


}


int main() {
  using namespace kaldi;
  kaldi::CuTensorUnitTest<double>();
  kaldi::CuTensorUnitTest<float>();
  KALDI_LOG << "Tests succeeded.\n";
}
