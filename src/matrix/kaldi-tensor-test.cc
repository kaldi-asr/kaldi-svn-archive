// matrix/kaldi-tensor-test.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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
  typedef std::pair<int32, int32> DS;
  
  {
    Matrix<Real> m(1, 90);
    std::vector<DS> dims_strides;
    dims_strides.push_back(DS(10, 1));
    dims_strides.push_back(DS(9, 10));
    
    Tensor<Real> t(dims_strides, m);

    t.Flatten();
    KALDI_ASSERT(t.NumIndexes() == 1 &&
                 t.Dim(0) == 90 && t.Stride(0) == 1 &&
                 t.Data() == m.Data());
  }

  {
    Matrix<Real> m(1, 90);
    std::vector<DS> dims_strides;
    dims_strides.push_back(DS(30, 1));
    dims_strides.push_back(DS(30, 1));
    dims_strides.push_back(DS(30, 1));
    

    Tensor<Real> t(dims_strides, m);
    t.Flatten();
    KALDI_ASSERT(t.NumIndexes() == 1 &&
                 t.Dim(0) == 90 && t.Stride(0) == 1 &&
                 t.Data() == m.Data());
  }


  {
    Matrix<Real> m(1, 90);
    std::vector<DS> dims_strides;
    dims_strides.push_back(DS(3, 1));
    dims_strides.push_back(DS(3, 1));
    dims_strides.push_back(DS(10, 9));
    dims_strides.push_back(DS(3, 1));
    
    Tensor<Real> t(dims_strides, m);
    t.Flatten();
    KALDI_ASSERT(t.NumIndexes() == 1 &&
                 t.Dim(0) == 90 && t.Stride(0) == 1 &&
                 t.Data() == m.Data());
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
  
  
}


template<class Real>
void TensorUnitTest() {
  TestFirstDimOverlaps();
  TestFlatten<Real>();
}


}


int main() {
  using namespace kaldi;
  kaldi::TensorUnitTest<double>();
  kaldi::TensorUnitTest<float>();
  KALDI_LOG << "Tests succeeded.\n";
}

