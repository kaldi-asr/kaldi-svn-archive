// matrix/kaldi-tensor-test.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2014  Pegah Ghahremani
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

namespace kaldi {

// This function computes 2D convolution of matrix m1 and m2
// This function handles the case where any of two dimensions sum up to (the other one + 1)
// it computes component (i,j) of m3 as a 2D convolution of block (i,j) to (i+row2, j+col2)
// from matrix m1 with matrix m2. 2D convolution of two blocks with same size is
// sum of product of their corresponding elements.
// e.g. 2D-Convolution of 2X2 matrix A and B is A(0,0) * B(0,0) + A(1,0) + B(1,0) +
// A(0,1) * B(0,1) + A(1,1) * B(1,1)
template<typename Real> 
void Simple2DConvolution(const Matrix<Real> &m1, const Matrix<Real> &m2,
                         Matrix<Real> *m3) {
  int32 col1 = m1.NumCols(), col2 = m2.NumCols(), col3 = m3->NumCols(),
    row1 = m1.NumRows(), row2 = m2.NumRows(), row3 = m3->NumRows();
  KALDI_ASSERT(((col1 + 1 == col2 + col3 && row1 + 1 == row2 + row3) ||
    (col3 + 1 == col1 + col2 && row3 + 1 == row1 + row2) || 
    (col2 + 1 == col1 + col3 && row2 + 1 == row1 + row3)));

  if (col1 + 1 == col2 + col3 && row1 + 1 == row2 + row3) {
    for (int i = 0; i < row3; i++) {
      for(int j = 0; j < col3; j++) {
        Real sum = static_cast<Real>(0.0);
        for (int k = 0; k < row2; k++) 
          for (int l =  0; l < col2; l++)
            sum += m1(i + k, j + l) * m2(k , l);
        (*m3)(i, j) = sum; 
      }
    }
  } else if (col2 + 1 == col1 + col3 && row2 + 1 == row1 + row3) {
    for (int i = 0; i < row3; i++) {
      for (int j = 0; j < col3; j++) {
        Real sum = static_cast<Real>(0.0);    
        for (int k = 0; k < row1; k++) 
          for (int l = 0; l < col1; l++) 
            sum += m2(i + k, j + l) * m1(k , l);  
        (*m3)(i, j) = sum;
      }
    }
  } else if (col3 + 1 == col1 + col2 && row3 + 1 == row1 + row2) {
    for (int i = 0; i < row3; i++) {
      for (int j = 0; j < col3; j++) {
        Real sum =0.0;
        for (int k = std::max(i - row2 + 1, 0); k < std::min(i+1, row1); k++)
          for (int l = std::max(j - col2 + 1, 0); l < std::min(j+1, col1); l++)
            sum += m1(k,l) * m2(i - k, j - l);
        (*m3)(i, j) = sum;
      }
    }
  }
}

void TestFirstDimOverlaps() {
  std::vector<TensorOperationDims> dims_vec(5);
  TensorOperationDims *dims_ptr = &(dims_vec[0]);
  dims_vec[0].dim = 10;
  dims_vec[0].stride_c = 0;  
  KALDI_ASSERT(FirstDimOverlaps(1, dims_ptr)); // due to zero stride.
  dims_vec[0].stride_c = 1;
  KALDI_ASSERT(!FirstDimOverlaps(1, dims_ptr));

  dims_vec[1].dim = 10;
  dims_vec[1].stride_c = 10;
  KALDI_ASSERT(!FirstDimOverlaps(2, dims_ptr));  

  dims_vec[1].dim = 10;
  dims_vec[1].stride_c = 9;
  KALDI_ASSERT(FirstDimOverlaps(2, dims_ptr)); // since 9 < 10.

  dims_vec[1].stride_c = 100;
  dims_vec[0].stride_c = 10;
  KALDI_ASSERT(!FirstDimOverlaps(2, dims_ptr));

  dims_vec[1].stride_c = 1;
  KALDI_ASSERT(!FirstDimOverlaps(2, dims_ptr));

  dims_vec[1].stride_c = 2;
  KALDI_ASSERT(FirstDimOverlaps(2, dims_ptr));
  dims_vec[1].stride_c = 1;
  KALDI_ASSERT(!FirstDimOverlaps(2, dims_ptr));  

  dims_vec[2].dim = 2;
  dims_vec[2].stride_c = 1;
  KALDI_ASSERT(FirstDimOverlaps(3, dims_ptr));  

  dims_vec[1].dim = 9;
  KALDI_ASSERT(!FirstDimOverlaps(3, dims_ptr));  
}

template<class Real>
void TestFlatten() {
  typedef std::pair<int32, int32> DimsStrides;
  
  {
    Matrix<Real> m(1, 90);
    std::vector<DimsStrides> dims_strides;
    dims_strides.push_back(DimsStrides(10, 1));
    dims_strides.push_back(DimsStrides(9, 10));
    
    Tensor<Real> t(dims_strides, m);

    t.Flatten();
    KALDI_ASSERT(t.NumIndexes() == 1 &&
                 t.Dim(0) == 90 && t.Stride(0) == 1 &&
                 t.Data() == m.Data());
  }

  {
    Matrix<Real> m(1, 90);
    std::vector<DimsStrides> dims_strides;
    dims_strides.push_back(DimsStrides(30, 1));
    dims_strides.push_back(DimsStrides(30, 1));
    dims_strides.push_back(DimsStrides(30, 1));
    

    Tensor<Real> t(dims_strides, m);
    t.Flatten();
    KALDI_ASSERT(t.NumIndexes() == 1 &&
                 t.Dim(0) == 90 && t.Stride(0) == 1 &&
                 t.Data() == m.Data());
  }


  {
    Matrix<Real> m(1, 90);
    std::vector<DimsStrides> dims_strides;
    dims_strides.push_back(DimsStrides(3, 1));
    dims_strides.push_back(DimsStrides(3, 1));
    dims_strides.push_back(DimsStrides(10, 9));
    dims_strides.push_back(DimsStrides(3, 1));
    
    Tensor<Real> t(dims_strides, m);
    t.Flatten();
    KALDI_ASSERT(t.NumIndexes() == 1 &&
                 t.Dim(0) == 90 && t.Stride(0) == 1 &&
                 t.Data() == m.Data());
  }
}

template<typename Real>
void TestScale() {
  KALDI_LOG << "=== TestScale() ===\n";
  typedef std::pair<int32, int32> DimsStrides;
  { 
    // Scale elements of 2D-tensor by its sum to have unit sum.
    KALDI_LOG << "Scale 2D-tensor ";
    for (int t = 0; t < 10; t++) {
      int n = 20 + rand() % 10, m = 40 + rand() % 10;
      std::vector<DimsStrides> dims_strides;
      dims_strides.push_back(DimsStrides(n, 1));
      dims_strides.push_back(DimsStrides(m, n));
      Matrix<Real> m1(1, n * m);
      m1.SetRandn();
      Tensor<Real> t1(dims_strides, m1);
      Real norm1 = t1.Sum();
      t1.Scale(1.0/norm1);
      Real norm2 = t1.Sum();
      KALDI_LOG << "vector sum .vs. sum after scaling " << norm1 << " " << norm2;
      AssertEqual(norm2, static_cast<Real>(1.0));
    } 
  }
  {
    // Scale elements of 3D-tensor using its 2-norm to have unit norm.
    KALDI_LOG << " Scale 3D-tensor ";
    for (int t = 0; t < 10; t++) {
      int n = 20 + rand() % 10, m = 40 + rand() % 10,
        k = 10 + rand() % 5;
      std::vector<DimsStrides> dims_strides;
      dims_strides.push_back(DimsStrides(n, 1));
      dims_strides.push_back(DimsStrides(m, 1));
      dims_strides.push_back(DimsStrides(k, n * m));
     
      Matrix<Real> m1(1, n * m * k);
      m1.SetRandn();
      Tensor<Real> t1(dims_strides, m1);
      Real norm1 = t1.FrobeniusNorm();
      t1.Scale(1.0/norm1);
      Real norm2 = t1.FrobeniusNorm();
      KALDI_LOG << "2-norm of vector .vs. 2-norm after scaling " << norm1 << " " << norm2;
      AssertEqual(norm2, static_cast<Real>(1.0));
    }
  }
}
template<class Real>
void RandomizeTensorIndexes(Tensor<Real> *t1,
                            Tensor<Real> *t2) {
  // First make sure the number of indices match.
  int32 max_num_indices = std::max(t1->NumIndices(),
                                   t2->NumIndices());
  if (t1->NumIndices != max_num_indices)
    *t1 = Tensor<Real>(max_num_indices, *t1);
  if (t2->NumIndices != max_num_indices)
    *t2 = Tensor<Real>(max_num_indices, *t2);
  
}

template<class Real>
void TestCopyFromTensor() {
 typedef std::pair<int32, int32> DimsStrides;
 Matrix<Real> m(1,150), m2(1,150);
 m.SetRandn();
 std::vector<DimsStrides> dims_strides;
 dims_strides.push_back(DimsStrides(10,1));
 dims_strides.push_back(DimsStrides(3,10));
 dims_strides.push_back(DimsStrides(5,30));  
 Tensor<Real> t(dims_strides, m);
 
 Tensor<Real> t2(dims_strides, m2);
 t2.CopyFromTensor(t);

}

template<typename Real>
void TestAddTensor() {
  KALDI_LOG << "=== TestAddTensor() ===\n";  
  typedef std::pair<int32, int32> DimsStrides;
  {
    // Sum elements of a Tensor
    for (int t = 0; t < 10; t++) {
      int32 n = 20 + rand() % 10, m = 40 + rand() % 20;
      std::vector<DimsStrides> A_dims_strides, B_dims_strides;
      A_dims_strides.push_back(DimsStrides(n,1));
      A_dims_strides.push_back(DimsStrides(m,n));
      Matrix<Real> A_mat(1, n * m);
      A_mat.SetRandn();
      //A_mat.Set(static_cast<Real>(1));
      Real A_sum = A_mat.Sum();
      Tensor<Real> A_tensor(A_dims_strides, A_mat);
      
      B_dims_strides.push_back(DimsStrides(1,1));
      B_dims_strides.push_back(DimsStrides(1,1));
      Matrix<Real> B_mat(1, 1);
      Tensor<Real> B_tensor(B_dims_strides, B_mat);
       
      B_tensor.AddTensor(1.0, A_tensor);
      Real* B_sum = B_tensor.Data();
      Real sum = A_tensor.Sum();

      KALDI_LOG << " Asum vs. Bsum vs. tensor sum " << A_sum << " vs. " << B_sum[0]
                << " .vs. " << sum;
      AssertEqual(A_sum, sum);
      AssertEqual(A_sum, B_sum[0]);
    }
  }
}

template<typename Real>
void TestAddTensorTensor() {
  KALDI_LOG << "=== TestAddTensorTensor() ===\n";
  typedef std::pair<int32, int32> DimsStrides;
  {
    for (int t = 0; t < 20; t++) { 
      // matrix multiplication using tensor product
      int n = 50 + rand() % 20, m = 50 + rand() % 20, k = 100 + rand() % 40;
      std::vector<DimsStrides> A_dims_strides, B_dims_strides,
        C_dims_strides;
      A_dims_strides.push_back(DimsStrides(n,1));
      A_dims_strides.push_back(DimsStrides(k,n));
      A_dims_strides.push_back(DimsStrides(1,n*k));
      B_dims_strides.push_back(DimsStrides(1,1));
      B_dims_strides.push_back(DimsStrides(k,1));
      B_dims_strides.push_back(DimsStrides(m,k));
      C_dims_strides.push_back(DimsStrides(n,1));
      C_dims_strides.push_back(DimsStrides(1,n));
      C_dims_strides.push_back(DimsStrides(m,n));
      
      Matrix<Real> A_mat(1, n * k), B_mat(1, m * k),
        C_mat(1, n * m), 
        A_mat2(n , k), B_mat2(k, m),
        C_mat2(m , n);
      A_mat.SetRandn(); B_mat.SetRandn();
      for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
          A_mat2(i,j) = A_mat(0,n * j + i);
      for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
          B_mat2(i,j) = B_mat(0, k * j + i);
      C_mat2.AddMatMat(1.0, B_mat2, kTrans, A_mat2, kTrans, 0.0);
      
      Tensor<Real> A_tensor(A_dims_strides, A_mat),
        B_tensor(B_dims_strides, B_mat),
        C_tensor(C_dims_strides, C_mat);
      C_tensor.AddTensorTensor(1.0, A_tensor, B_tensor, 0.0);
      Real Cmat_sum = C_mat2.Sum();
      Real Ctensor_sum = C_tensor.Sum();
      KALDI_LOG << " Cmat_sum ,  Ctensor_sum " << Cmat_sum << " .vs. " <<  Ctensor_sum;
      AssertEqual(Cmat_sum, Ctensor_sum);
    }
  }
  {
    // Compute 2-norm of a 2D tensor, which is the sqrt of sum of square its elements 
    for (int t = 0; t < 10; t++) {
      int32 n = 50 + rand() % 20, m = 20 + rand() % 10;
      Matrix<Real> A_mat(1, n * m), norm(1,1);
      A_mat.SetRandn();
      std::vector<DimsStrides> A_dims_strides, norm_dims_strides;
      A_dims_strides.push_back(DimsStrides(n,1));
      A_dims_strides.push_back(DimsStrides(m,n));
      norm_dims_strides.push_back(DimsStrides(1,1));

      Tensor<Real> A_tensor(A_dims_strides, A_mat), norm_t(norm_dims_strides, norm);
      norm_t.AddTensorTensor(static_cast<Real>(1.0), A_tensor, A_tensor, 
                           static_cast<Real>(0.0));
      Real norm1 = A_mat.FrobeniusNorm(),
        norm2 = std::sqrt(*norm_t.Data()),
        norm3 = A_tensor.FrobeniusNorm();
      KALDI_LOG << " 2-norm vs. 2D-tensor 2-norm, 2D-tensor FrobeniusNorm " << norm1 << " " << norm2 << " " << norm3;
      AssertEqual(norm1, norm2);
      AssertEqual(norm1, norm3);
    }
  }
  {
    // Compute 2-norm of a 4D tensor, which is the sqrt of sum of square its elements 
    for (int t = 0; t < 10; t++) {
      int32 n = 50 + rand() % 20, m = 20 + rand() % 10,
        k = 20 + rand() % 10, p = 40 + rand() % 20;
      Matrix<Real> A_mat(1, n * m * k * p), norm(1,1);
      A_mat.SetRandn();
      std::vector<DimsStrides> A_dims_strides, norm_dims_strides;
      A_dims_strides.push_back(DimsStrides(n,1));
      A_dims_strides.push_back(DimsStrides(m,n));
      A_dims_strides.push_back(DimsStrides(k, m * n));
      A_dims_strides.push_back(DimsStrides(p, m * n *k));
      norm_dims_strides.push_back(DimsStrides(1,1));

      Tensor<Real> A_tensor(A_dims_strides, A_mat), norm_t(norm_dims_strides, norm);
      norm_t.AddTensorTensor(static_cast<Real>(1.0), A_tensor, A_tensor, 
                           static_cast<Real>(0.0));
      Real norm1 = A_mat.FrobeniusNorm(),
        norm2 = std::sqrt(*norm_t.Data()),
        norm3 = A_tensor.FrobeniusNorm();
      KALDI_LOG << " real 2-norm vs. 4D-tensor 2-norm, 4D-tensor FrobeniusNorm " << norm1 << " " << norm2 << " " << norm3;
      AssertEqual(norm1, norm2);
      AssertEqual(norm1, norm3);
    }
  }
  {
    // The simplest case where for all index i,
    // t1.Dim(i) = t2.Dim(i) = t3.Dim(i)
    // check if AddTensorTensor(A, B) = AddTensorTensor(B, A)
    std::vector<DimsStrides> dims_strides;
    dims_strides.push_back(DimsStrides(3,1));
    dims_strides.push_back(DimsStrides(4,3));
    dims_strides.push_back(DimsStrides(5,12));

    Matrix<Real> m1(1, 60), m2(1, 60), m3(1, 60),m4(1, 60);
    m2.SetRandn();
    m3.AddMat(-1.0, m2);
    Tensor<Real> t1(dims_strides, m1), t2(dims_strides, m2),
      t3(dims_strides, m3), t4(dims_strides, m4);
    t1.AddTensorTensor(static_cast<Real>(1.0), t2, t3, static_cast<Real>(0.0));
    t4.AddTensorTensor(static_cast<Real>(1.0), t3, t2, static_cast<Real>(0.0));
    KALDI_ASSERT(t1.ApproxEqual(t4));
  }

  {
    // The case where t1.Dim(i) = t2.Dim(i),
    // but t3.Dim(i) = 1 for some i;
    // check if AddTensorTensor(A, B) = AddTensorTensor(B, A)
    int32 n = 3, m = 4, k = 10, p = 5; 
    std::vector<DimsStrides> dims_strides1;
    dims_strides1.push_back(DimsStrides(n, 1));
    dims_strides1.push_back(DimsStrides(m, n));
    dims_strides1.push_back(DimsStrides(k, n * m));
    dims_strides1.push_back(DimsStrides(p, n * m * k));
    
    std::vector<DimsStrides> dims_strides2;
    dims_strides2.push_back(DimsStrides(n, 1));
    dims_strides2.push_back(DimsStrides(1, n));
    dims_strides2.push_back(DimsStrides(k, n));
    dims_strides2.push_back(DimsStrides(1, k * n));
    
    Matrix<Real> m1(1, n * m * k * p), m2(1, n * m * k *p), 
      m3(1, n * k), m4(1, n * m * k * p);
    m2.SetRandn(); m3.SetRandn(); 
    Tensor<Real> t1(dims_strides1, m1), t2(dims_strides1, m2),
      t3(dims_strides2, m3), t4(dims_strides1, m4);
    t1.AddTensorTensor(static_cast<Real>(1), t2, t3, static_cast<Real>(0.0));
    t4.AddTensorTensor(static_cast<Real>(1), t3, t2, static_cast<Real>(0.0));
    KALDI_LOG << " norm2 t1 , t4 " << t1.FrobeniusNorm() << " " << t4.FrobeniusNorm();
    KALDI_ASSERT(t1.ApproxEqual(t4));
  }
}
template<typename Real> 
void TestConvTensorTensor() {
  KALDI_LOG << "=== TestConvTensorTensor() ===\n";
  typedef std::pair<int32, int32> DimsStrides;
  {
    for (int t = 0; t < 10; t++) {
      int b0 = 12 + rand() % 4, b1 = 4 + rand() % 3, b2 = 1 , b3 = 12 + rand() % 4,
        c0 = 8 + rand() % 2, c1 = b1, c2 = 4 + rand() % 2, c3 = 8 + rand() % 2,
        a0 = b0 - c0 + 1, a1 = c1, a2 = c2, a3 = b3 - c3 + 1,
        n = 128 + rand() % 10;
      Real scalar = RandGauss();
      std::vector<DimsStrides> A_dims_strides, B_dims_strides, C_dims_strides;
      B_dims_strides.push_back(DimsStrides(n, b0 * b1 * b2 * b3));
      B_dims_strides.push_back(DimsStrides(b0,b1 * b2 * b3)); 
      B_dims_strides.push_back(DimsStrides(b1, b2 * b3));
      B_dims_strides.push_back(DimsStrides(b2, 0));
      B_dims_strides.push_back(DimsStrides(b3, 1));

      C_dims_strides.push_back(DimsStrides(n, c0 * c1 * c2 * c3));
      C_dims_strides.push_back(DimsStrides(c0, c1 * c2 * c3));
      C_dims_strides.push_back(DimsStrides(c1, c2 * c3));
      C_dims_strides.push_back(DimsStrides(c2, c3));
      C_dims_strides.push_back(DimsStrides(c3, 1));
      
      A_dims_strides.push_back(DimsStrides(a0, a1 * a2 * a3));
      A_dims_strides.push_back(DimsStrides(a1, a2 * a3));
      A_dims_strides.push_back(DimsStrides(a2, a3));
      A_dims_strides.push_back(DimsStrides(a3, 1));

      for (int i = 0; i < 5; i++) {
        Matrix<Real> A_mat(1, a0 * a1 * a2 * a3),
          B_mat(1, n * b0 * b1 * b2 * b3),
          C_mat(1, n * c0 * c1 * c2 * c3);
        B_mat.Set(1.0);
        C_mat.Set(1.0);
        Tensor<Real> A_tensor(A_dims_strides, A_mat),
          B_tensor(B_dims_strides, B_mat),
          C_tensor(C_dims_strides, C_mat);
        Matrix<Real> tmp(1, a0 * a1 * a2 * a3);     
        A_tensor.ConvTensorTensor(scalar, B_tensor, C_tensor);
        tmp.Add(scalar * c0 * c3 * n);
        KALDI_ASSERT(A_mat.ApproxEqual(tmp));
      }
    }
  }
  {
    // test 2D convolution of 2D matrices
    // Check if ConvTensorTensor(t1, t2) = ConvTensorTensor(t2, t1)
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
                   C_mat(1, n3 * m3), C_mat2(1, n3 * m3);
      A_mat.SetRandn();
      B_mat.SetRandn();
      Tensor<Real> A_tensor(A_dims_strides, A_mat),
        B_tensor(B_dims_strides, B_mat),
        C_tensor(C_dims_strides, C_mat),
        C_tensor2(C_dims_strides, C_mat2);
      C_tensor.ConvTensorTensor(1.0, A_tensor, B_tensor);
      C_tensor2.ConvTensorTensor(1.0, B_tensor, A_tensor);
      C_tensor.ApproxEqual(C_tensor2);
      Real norm1 = C_tensor.FrobeniusNorm(), 
        norm2 = C_tensor2.FrobeniusNorm();
      KALDI_LOG << " Conv(t1, t2) vs. Conv(t2, t1) 2-norm  " << norm1 << " " << norm2;
      Matrix<Real> A_mat2(n1, m1), B_mat2(n2, m2), 
       C_mat3(n3, m3), C_mat4(n3, m3);
      for (int i = 0; i < n1; i++)
         for (int j = 0; j < m1; j++) 
           A_mat2(i, j) = A_mat(0, n1 * j + i);
      for (int i = 0; i < n2; i++)
         for (int j = 0; j < m2; j++)
           B_mat2(i, j) = B_mat(0, n2 * j + i);
      Simple2DConvolution(A_mat2, B_mat2, &C_mat3);
      Real *C_data = C_tensor.Data();
      for (int i = 0; i < n3; i++)   
         for (int j = 0; j < m3; j++) 
           C_mat4(i, j) = C_data[j * n3 + i];
      // compare convolution using ConvTensorTensor and simple2DConvolution 
      C_mat3.ApproxEqual(C_mat4);
    }
  }
  {
    // test 2D convolution of matrices by comparing simple 2D convolution
    // of 2 matrices .vs. tensor convolution of their corresponding 2D tensors.
    for (int t = 0; t < 10; t++) {
      int n2 = 20 + rand() % 10, m2 = 20 + rand() % 10, 
        n3 = 40 + rand() % 20, m3 = 20 + rand() % 10,
        n1 = n2 + n3 - 1, m1 = m2 + m3 - 1;
      std::vector<DimsStrides> A_dims_strides, B_dims_strides, C_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1,1));
      A_dims_strides.push_back(DimsStrides(m1,n1));

      B_dims_strides.push_back(DimsStrides(n2,1));
      B_dims_strides.push_back(DimsStrides(m2,n2));

      C_dims_strides.push_back(DimsStrides(n3,1));
      C_dims_strides.push_back(DimsStrides(m3,n3));

      Matrix<Real> A_mat(1, n1 * m1), B_mat(1, n2 * m2),
                   C_mat(1, n3 * m3), C_mat3(1, n3 * m3),
                   A_mat2(n1, m1), B_mat2(n2, m2),
                   C_mat2(n3, m3);
      A_mat.SetRandn();
      B_mat.SetRandn();
      for (int i = 0; i < n1; i++)
        for (int j = 0; j < m1; j++) 
          A_mat2(i, j) = A_mat(0, n1 * j + i);
      for (int i = 0; i < n2; i++) 
        for (int j = 0; j < m2; j++) 
          B_mat2(i, j) = B_mat(0, n2 * j + i);
      Tensor<Real> A_tensor(A_dims_strides, A_mat),
        B_tensor(B_dims_strides, B_mat),
        C_tensor(C_dims_strides, C_mat),
        C_tensor2(C_dims_strides, C_mat3);
      // d1 + 1 = d2 + d3
      // t3.ConTensorTensor(t1,t2)
      Simple2DConvolution(A_mat2, B_mat2, &C_mat2);
      C_tensor.ConvTensorTensor(1.0, A_tensor, B_tensor);
      Real *C_data = C_tensor.Data();
      Matrix<Real> tmp(n3, m3);
      for (int i = 0; i < n3; i++)
        for (int j = 0; j < m3; j++)
          tmp(i,j) = C_data[j * n3 + i];
      
      Real norm1 = C_tensor.FrobeniusNorm(), 
        norm2 = C_mat2.FrobeniusNorm();
      KALDI_LOG << " A(" << n1 << ", " << m1 << ") , B( " << n2 << ", " << m2 << ") , C(" << n3 << ", " << m3 <<")";
      KALDI_LOG << " simple 2D convolution norm .vs. Tensor Convolution norm " <<  norm1 << " " << norm2;
      AssertEqual(norm1, norm2);
      AssertEqual(tmp, C_mat2);
   
      //d2 + 1 = d1 + d3
      //t3.ConTensorTensor(t2,t1)
      Simple2DConvolution(B_mat2, A_mat2, &C_mat2);
      C_tensor2.ConvTensorTensor(1.0, B_tensor, A_tensor);
      norm1 = C_tensor2.FrobeniusNorm();
      norm2 = C_mat2.FrobeniusNorm();
      KALDI_LOG << " simple 2D convolution norm .vs. Tensor Convolution norm " <<  norm1 << " " << norm2;
      C_data = C_tensor.Data();
      for (int i = 0; i < n3; i++)
        for (int j = 0; j < m3; j++)
          tmp(i,j) = C_data[j * n3 + i];
      AssertEqual(norm1, norm2);
      AssertEqual(tmp, C_mat2);
    }
  }
}
template<typename Real> 
void TestMinMax() {
  typedef std::pair<int32, int32> DimsStrides;   
  {
    for (int i = 0; i < 10; i++) {
      int n1 = 20 + rand() % 10, m1 = 30 + rand() % 10;
      std::vector<DimsStrides> A_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1,1));  
      A_dims_strides.push_back(DimsStrides(m1,n1));  
      Matrix<Real> A_mat(1, n1 * m1);
      A_mat.SetRandn();
      Tensor<Real> A_tensor(A_dims_strides, A_mat);
      AssertEqual(A_mat.Max(), A_tensor.Max());
      AssertEqual(A_mat.Min(), A_tensor.Min());
      KALDI_LOG << " Tensor Max .vs. true Max " << A_tensor.Max() << " " << A_mat.Max(); 
      KALDI_LOG << " Tensor Min .vs. true Min " << A_tensor.Min() << " " << A_mat.Min();
    }
  }
  {
    for (int i = 0; i < 10; i++) {
      int n1 = 20 + rand() % 10, n2 = 30 + rand() % 10,
        n3 = 10 + rand() % 5;
      std::vector<DimsStrides> A_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1, 1));
      A_dims_strides.push_back(DimsStrides(1, 0));
      A_dims_strides.push_back(DimsStrides(n2, n1));
      A_dims_strides.push_back(DimsStrides(n3, n1 * n2));
      Matrix<Real> A_mat(1, n1 * n2 * n3);
      A_mat.SetRandn();
      Tensor<Real> A_tensor(A_dims_strides, A_mat);
      AssertEqual(A_mat.Max(), A_tensor.Max());
      AssertEqual(A_mat.Min(), A_tensor.Min());
      KALDI_LOG << " Tensor Max .vs. true Max " << A_tensor.Max() << " " << A_mat.Max(); 
      KALDI_LOG << " Tensor Min .vs. true Min " << A_tensor.Min() << " " << A_mat.Min();
    }
  }
}
template<class Real>
void TestApplyPow() {
  typedef std::pair<int32, int32> DimsStrides;
  {
    for (int i = 0; i < 10; i++) {  
      int n1 = 20 + rand() % 10, m1 = 30 + rand() % 10; 
      std::vector<DimsStrides> A_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1,1));  
      A_dims_strides.push_back(DimsStrides(m1,n1));  
      Matrix<Real> A_mat(1, n1 * m1);
      A_mat.SetRandn();
      A_mat.Add(std::abs(2 * A_mat.Min()));
      Matrix<Real> B_mat(A_mat);
      Tensor<Real> A_tensor(A_dims_strides, A_mat);
      Real power = 2 + RandUniform();
      A_tensor.ApplyPow(3.1);
      B_mat.ApplyPow(3.1);
      Real tmp1 = A_tensor.Sum(), tmp2 = B_mat.Sum();
      AssertEqual(tmp1, tmp2);
      KALDI_LOG << " power .vs. tmp1 .vs. tmp2 " << power << " " << tmp1 << " " << tmp2;
    }
  }
  {
    for (int i = 0; i < 10; i++) { 
      int n1 = 20 + rand() % 10, n2 = 30 + rand() % 10,
        n3 = 10 + rand() % 5; 
      std::vector<DimsStrides> A_dims_strides,
        B_dims_strides;
      A_dims_strides.push_back(DimsStrides(n1,1));  
      A_dims_strides.push_back(DimsStrides(n2,n1));  
      Matrix<Real> A0(1, n1 * n2), A1_mat(1, n1 * n2),
        A2_mat(1, n1 * n2);
      A0.SetRandn();
      A0.MulElements(A0);

      A1_mat.CopyFromMat(A0);
      Tensor<Real> A_tensor(A_dims_strides, A1_mat); 
      A_tensor.ApplyPow(0.5);
      A_tensor.ApplyPow(2.0);
      AssertEqual(A1_mat, A0);

      A2_mat.CopyFromMat(A0);
      Tensor<Real> A2_tensor(A_dims_strides, A2_mat);
      A2_tensor.ApplyPow(1.0/3.0);
      A2_tensor.ApplyPow(3.0);
      AssertEqual(A2_mat, A0);

      //
      B_dims_strides.push_back(DimsStrides(n1, 1));
      B_dims_strides.push_back(DimsStrides(1, 0));
      B_dims_strides.push_back(DimsStrides(n2, n1));
      B_dims_strides.push_back(DimsStrides(n3, n1 * n2));
      Matrix<Real> B(1, n1 * n2 * n3), B1_mat(1, n1 * n2 * n3);
      B.SetRandn();
      B.MulElements(B);
      B1_mat.CopyFromMat(B);
      Tensor<Real> B1_tensor(B_dims_strides, B1_mat);
      B1_tensor.ApplyPow(0.5);
      B1_tensor.ApplyPow(2.0);
      AssertEqual(B1_mat, B);

      //
    }
  }
}
template<class Real>
void TensorUnitTest() {
  TestFirstDimOverlaps();
  TestFlatten<Real>();
  TestScale<Real>();
  TestCopyFromTensor<Real>();
  TestAddTensor<Real>();
  TestAddTensorTensor<Real>();
  TestConvTensorTensor<Real>();
  TestMinMax<Real>(); 
  TestApplyPow<Real>();
}


}


int main() {
  using namespace kaldi;
  kaldi::TensorUnitTest<double>();
  kaldi::TensorUnitTest<float>();
  KALDI_LOG << "Tests succeeded.\n";
}

