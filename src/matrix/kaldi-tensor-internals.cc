// matrix/kaldi-tensor-internals.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include "matrix/kaldi-tensor-internals.h"
#include "matrix/cblas-wrappers.h"

namespace kaldi {


/// This reorders the dims given a provided order.  
void ReorderTensorDims(const std::vector<size_t> &order,
                       std::vector<TensorOperationDims> *vec) {
  std::vector<TensorOperationDims> vec_copy(*vec);
  size_t size = order.size();
  KALDI_ASSERT(size == vec->size());
  for (size_t i = 0; i < size; i++) {
    KALDI_ASSERT(order[i] < size);
    (*vec)[i] = vec_copy[order[i]];
    // the next two lines are a self-check, that 'order' doesn't have
    // duplicates.
    KALDI_ASSERT(vec_copy[order[i]].dim != -1);
    vec_copy[order[i]].dim = -1;
  }
}

void ExciseUnitDims(std::vector<TensorOperationDims> *vec) {
  for (size_t i = 0; i < vec->size(); i++)
    if ((*vec)[i].dim == 1)
      vec->erase(vec->begin() + i);
}


void ReorderSumToEnd(std::vector<TensorOperationDims> *vec) {
  std::vector<size_t> order;
  size_t size = vec->size();
  order.reserve(size);
  
  for (int32 iter = 1; iter <= 2; iter++) {
    for (size_t i = 0; i < size; i++) {
      bool is_summed = ((*vec)[i].stride_c == 0 && (*vec)[i].dim != 1);
      // wait till iteration 2 to handle the summed dimensions.
      if (is_summed == (iter == 2))
        order.push_back(i);
    }
  }
  ReorderTensorDims(order, vec);
}        

inline bool HasNegativeStrides(const TensorOperationDims &dims) {
  return (dims.stride_a < 0 || dims.stride_b < 0 || dims.stride_c < 0);
}


template<class Real>
void AddTensorTensorOrder1(const TensorOperationDims *dims,
                           Real alpha,
                           const Real *a_data,
                           const Real *b_data,
                           Real *c_data,
                           Real beta) {
  TensorOperationDims dims0 = dims[0];

  if (HasNegativeStrides(dims0)) {
    // BLAS won't deal with negative strides, I think.  We just
    // handle it manually if strides are negative.
    if (dims0.stride_c == 0) {
      *c_data *= beta;
      for (int32 i = 0; i < dims0.dim;
           i++, a_data += dims0.stride_a, b_data += dims0.stride_b)
        *c_data += alpha * *a_data * *b_data;
    } else {
      for (int32 i = 0; i < dims0.dim;
           i++, a_data += dims0.stride_a, b_data += dims0.stride_b,
               c_data += dims0.stride_c)
        *c_data = beta * *c_data + alpha * *a_data * *b_data;
    }
    return;
  }
  
  
  if (dims0.stride_a != 0) {
    if (dims0.stride_b != 0) {
      if (dims0.stride_c != 0) {
        // Elementwise multiplication of vectors:
        // c[i] = alpha * a[i] * b[i] + beta * c[i].
        // We can accomplish this using {s,d}gbmv, which is band-diagonal
        // matrix-vector multiplication, setting the "num_above" and "num_below"
        // parameters to zero.
        cblas_Xgbmv(kNoTrans, dims0.dim, dims0.dim, 0, 0, alpha,
                    a_data, dims0.stride_a, b_data, dims0.stride_b,
                    beta, c_data, dims0.stride_c);
      } else {
        // dot product
        *c_data = beta * *c_data +
            alpha * cblas_Xdot(dims0.dim,
                       a_data, dims0.stride_a,
                       b_data, dims0.stride_b);
      }
    } else { // stride_b == 0
      if (dims0.stride_c != 0) {
        // Adding a vector to a vector, c += (alpha * b[0]) * a + beta * c.
        if (beta != 1.0)
          cblas_Xscal(dims0.dim, beta, c_data, dims0.stride_c);
        cblas_Xaxpy(dims0.dim, alpha * b_data[0],
                    a_data, dims0.stride_a,
                    c_data, dims0.stride_c);
      } else {
        // sum the vector a, times scalar b; add to scalar c.
        *c_data = beta * *c_data +
            alpha * *b_data * Xsum(dims0.dim, a_data, dims0.stride_a);
      }
    }
  } else { // stride_a == 0
    if (dims0.stride_b != 0) {
      if (dims0.stride_c != 0) {
        // Adding a vector to a vector, c += (alpha * a[0]) * b + beta * c.
        if (beta != 1.0)
          cblas_Xscal(dims0.dim, beta, c_data, dims0.stride_c);
        cblas_Xaxpy(dims0.dim, alpha * a_data[0],
                    b_data, dims0.stride_b,
                    c_data, dims0.stride_c);
      } else {
        // sum the vector b, times scalar a; add to scalar c.
        *c_data = beta * *c_data +
            alpha * *a_data * Xsum(dims0.dim, b_data, dims0.stride_b);
      }
    } else { // stride_b == 0
      if (dims0.stride_c != 0) {
        // adding scalar a * scalar b to all elements of c.
        double additive_term = alpha * *a_data * *b_data;
        int32 d = dims0.dim;
        if (beta != 0) {
          for (int32 i = 0; i < d; i++)
            c_data[i] = additive_term;
        } else {
          for (int32 i = 0; i < d; i++)
            c_data[i] = beta * c_data[i] + additive_term;
        }
      } else {
        // all are scalars.
        *c_data = alpha * *a_data * *b_data + beta * *c_data;
      }
    }
  }
}


// Returns true if overlap is detected for order-2 operations.  An example of
// overlap is dim0 = 2, stride0 = 9, dim1 = 10, stride1 = 1, which a 2 x 10
// matrix with a stride of 9, which is too little (should be >= 10).
// This won't return true just because one of the strides is zero,
// or because a stride is negative.
inline bool HasOverlapOrder2(int32 dim0, int32 stride0, int32 dim1, int32 stride1) {
  bool ok = (stride0 >= dim1 * stride1 || stride1 >= dim0 * stride0);
  return !ok;
}

inline bool HasOverlapOrder2(const TensorOperationDims *dims) {
  // This is called after we excised unit dims, so neither
  // dim should equal one.
  KALDI_ASSERT(dims[0].dim != 1 && dims[1].dim != 1);
  return HasOverlapOrder2(dims[0].dim, dims[0].stride_a,
                          dims[1].dim, dims[1].stride_a) ||
      HasOverlapOrder2(dims[0].dim, dims[0].stride_b,
                       dims[1].dim, dims[1].stride_b) ||
      HasOverlapOrder2(dims[0].dim, dims[0].stride_c,
                       dims[1].dim, dims[1].stride_c);
}


// Returns true if these dims and strides correspond to a valid matrix.
// Note: none of the dims can be one as we already excised unit dims.
// A valid matrix would have one unit stride and one greater than one,
// and no overlap.
bool IsValidMatrixOrder3(int32 dim0, int32 dim1, int32 dim2,
                         int32 stride0, int32 stride1, int32 stride2) {
  if (stride0 < 0 || stride1 < 0 || stride2 < 0)
    return false;
  // First swap so that stride0 > 1 and stride1 == 1, if possible.

  // First make it so that stride1 > 1, if possible.
  if (stride2 > 1) { std::swap(stride2, stride0); std::swap(dim2, dim0); }
  if (stride1 > 1) { std::swap(stride1, stride0); std::swap(dim1, dim0); }
  // Next make it so that stride1 == 1, if possible, and stride2 == 0
  if (stride2 == 1) { std::swap(stride2, stride1); std::swap(dim2, dim1); }
  if (stride0 <= 1 || stride1 != 1 || stride2 != 0) return false;
  if (stride0 < dim1) return false;  // Overlap between rows detected.
  return true;
}


/** This function tries to do the order-2 tensor operation using
    level-2 BLAS, and returns true if it was able to; otherwise
    it returns false.
 */
template<class Real>
inline bool AddTensorTensorOrder2Blas(const TensorOperationDims *dims,
                                      Real alpha,
                                      const Real *a_data,
                                      const Real *b_data,
                                      Real *c_data,
                                      Real beta) {
  if (HasOverlapOrder2(dims) || HasNegativeStrides(dims[0]) ||
      HasNegativeStrides(dims[1]))
    return false;
  
  TensorOperationDims dims0 = dims[0], dims1 = dims[1];
    
  if (dims1.stride_c > dims0.stride_c) {
    // Make it so the larger stride for the output is in dims0, so that
    // any matrix will be row-major and any vector will be accessed by
    // dim0.
    std::swap(dims1, dims0);
  }
  
  if (dims0.stride_c != 0 && dims1.stride_c == 1) {
    // The output c can be interpreted as a row-major matrix.
    
    if (dims0.stride_a == 0 && dims1.stride_a != 0 &&
        dims0.stride_b != 0 && dims1.stride_b == 0) {
      // c += b a^t.
      // Rather than duplicating code, we swap a and b and
      // use the code for c += b a^t below.
      std::swap(dims0.stride_a, dims0.stride_b);
      std::swap(dims1.stride_a, dims1.stride_b);
      std::swap(a_data, b_data);
    }

    // c += a b^t
    if (dims0.stride_a != 0 && dims1.stride_a == 0 &&
        dims0.stride_b == 0 && dims1.stride_b != 0) {
      if (beta != 1.0) {
        if (dims0.stride_c == dims1.dim) {
          // matrix C can be treated as one contigous vector
          cblas_Xscal(dims0.dim * dims1.dim, beta, c_data, 1);
        } else {
          for (int32 i = 0; i < dims0.dim; i++)
            cblas_Xscal(dims1.dim, beta, c_data + i * dims0.stride_c, 1);
        }
      }
      cblas_Xger(dims0.dim, dims1.dim, alpha, a_data, dims0.stride_a,
                 b_data, dims0.stride_b, c_data, dims0.stride_c);
      return true; // we're done. 
    }
  } else if (dims0.stride_c != 0 && dims1.stride_c == 0) {
    // The output c can be interpreted as a vector indexed by dim0.

    if (dims0.stride_b != 0 && dims1.stride_b != 0 && dims0.stride_a == 0) {
      // b is a matrix and a is a vector.  To avoid having to handle this
      // directly, we swap a and b and let it be handled below.
      std::swap(dims0.stride_a, dims0.stride_b);
      std::swap(dims1.stride_a, dims1.stride_b);
      std::swap(a_data, b_data);
    }
    
    if (dims0.stride_a != 0 && dims1.stride_a != 0 && dims0.stride_b == 0) {
      // a is a matrix and b is a vector.
      if (dims1.stride_a == 1) {  // dims0 corresponds to a's row.
        cblas_Xgemv(kNoTrans, dims0.dim, dims1.dim,
                    alpha, a_data, dims0.stride_a,
                    b_data, dims1.stride_b,
                    beta, c_data, dims0.stride_c);
        return true;
      } else if (dims0.stride_a == 1) {  // dims1 corresponds to a's row.
        cblas_Xgemv(kTrans, dims1.dim, dims0.dim,
                    alpha, a_data, dims1.stride_a,
                    b_data, dims1.stride_b,
                    beta, c_data, dims0.stride_c);
        return true;
      }
    }
    // There are a bunch of other special cases that we could try to handle
    // efficiently here, but we don't.  We will if we notice a problem.
  }
  return false;
}


template<class Real>
inline void AddTensorTensorOrder2(const TensorOperationDims *dims,
                                  Real alpha,
                                  const Real *a_data,
                                  const Real *b_data,
                                  Real *c_data,
                                  Real beta) {

  if (AddTensorTensorOrder2Blas(dims, alpha, a_data, b_data, c_data, beta))
    return;
  
  // We did not handle it above, so back off to generic code.

  // If c_data can be accessed in multiple distinct ways with different indexes,
  // anc dims[0].stride_c != 0, we need to scale c_data first, and set beta to
  // 1, or the result from the code below may not be correct.  The case where
  // dims[0].stride_c is easier to handle by setting beta to 1 after the 1st
  // iteration below.
  if (beta != 1.0 &&
      HasOverlapOrder2(dims[0].dim, dims[0].stride_c,
                       dims[1].dim, dims[1].stride_c)) {
    ScaleTensor(2, dims, beta, c_data);
    beta = 1.0;
  }
  
  // Just loop over the leading dimension.
  for (int32 i = 0; i < dims[0].dim; i++,
           a_data += dims[0].stride_a,
           b_data += dims[0].stride_b,
           c_data += dims[0].stride_c) {
    AddTensorTensorOrder1(dims + 1,
                          alpha, a_data,
                          b_data, c_data, beta);
    if (dims[0].stride_c == 0)
      beta = 1.0;
  }
}



inline bool AllValidMatricesOrder3(const TensorOperationDims *dims) {
  return IsValidMatrixOrder3(dims[0].dim, dims[1].dim, dims[2].dim,
                             dims[0].stride_a, dims[1].stride_a,
                             dims[2].stride_a) &&
       IsValidMatrixOrder3(dims[0].dim, dims[1].dim, dims[2].dim,
                           dims[0].stride_b, dims[1].stride_b,
                           dims[2].stride_b) &&
       IsValidMatrixOrder3(dims[0].dim, dims[1].dim, dims[2].dim,
                           dims[0].stride_c, dims[1].stride_c,
                           dims[2].stride_c);
  
}


/** This function tries to do the order-3 tensor operation using
    level-3 BLAS, and returns true if it was able to; otherwise
    it returns false.
 */
template<class Real>
inline bool AddTensorTensorOrder3Blas(const TensorOperationDims *dims,
                                      Real alpha,
                                      const Real *a_data,
                                      const Real *b_data,
                                      Real *c_data,
                                      Real beta) {
  if (!AllValidMatricesOrder3(dims))
    return false;
  
  TensorOperationDims dims0 = dims[0], dims1 = dims[1], dims2 = dims[2];
  
  // We will order the dimensions so the stride of c_data is (>1, 0, 1),
  // so that the first dimension is the row dimension, the second is summed over,
  // and the third is the column dimension.

  // First make it so dims0.stride_c > 1.
  if (dims1.stride_c > 1) std::swap(dims0, dims1);
  else if (dims2.stride_c > 1) std::swap(dims0, dims2);
  
  // Next make it so dims1.stride_c == 0.
  if (dims1.stride_c != 0) std::swap(dims1, dims2);

  KALDI_ASSERT(dims0.stride_c > 1 && dims1.stride_c == 0 && dims2.stride_c == 1);

  // Next make it so that matrix A is the one that has a nonzero stride for
  // index zero, by swapping A and B if necessary.
  if (dims0.stride_a == 0) {
    std::swap(a_data, b_data);
    std::swap(dims0.stride_a, dims0.stride_b);
    std::swap(dims1.stride_a, dims1.stride_b);
    std::swap(dims2.stride_a, dims2.stride_b);
  }

  // OK, now it's going to be a matrix multiplication (gemm)
  // C += alpha A B + beta C.

  // the following would be a code error if it failed.
  KALDI_ASSERT(dims0.stride_a == 1 || dims1.stride_a == 1);
  KALDI_ASSERT(dims1.stride_b == 1 || dims2.stride_b == 1);
  
  MatrixTransposeType transA = (dims0.stride_a == 1 ? kTrans : kNoTrans),
      transB = (dims1.stride_b == 1 ? kTrans : kNoTrans);
  // we're taking the max of one 1 and one >1, and get the >1.
  int32 stride_a = std::max(dims0.stride_a, dims1.stride_a),
      stride_b = std::max(dims1.stride_b, dims2.stride_b),
      stride_c = dims0.stride_c;

  cblas_Xgemm(transA, transB, dims0.dim, dims2.dim, dims1.dim,
              alpha, a_data, stride_a, b_data, stride_b,
              beta, c_data, stride_c);

  return true;
}


struct StrideGreater {
  inline bool operator () (const std::pair<int32, int32> &d1,
                           const std::pair<int32, int32> &d2) {
    return (std::abs(d1.second) > std::abs(d2.second));
  }
};


struct StrideLess {
  inline bool operator () (const std::pair<int32, int32> &d1,
                    const std::pair<int32, int32> &d2) {
    return (std::abs(d1.second) < std::abs(d2.second));
  }
};


bool FirstDimOverlaps(int32 num_indexes,
                      const TensorOperationDims *dims) {
  KALDI_ASSERT(num_indexes > 0 && dims[0].dim > 1);
  if (dims[0].stride_c == 0)
    return true;
  std::vector<std::pair<int32, int32> > dims_strides;
  dims_strides.reserve(num_indexes);

  int32 first_dim = dims[0].dim, first_abs_stride = std::abs(dims[0].stride_c);
  
  // range is the difference between the minimum and maximum offset in c_data from
  // considering those elements of dims_strides that have a stride less than the
  // leading dimension. 
  int32 range = 0;

  // min_larger_abs_stride is the smallest absolute stride other than for
  // index-position=0, that is not smaller than the first index-position's
  // absolute stride.
  int32 min_larger_abs_stride = -1;
  
  for (int32 i = 1; i < num_indexes; i++) {
    int32 dim = dims[i].dim, 
        abs_stride = std::abs(dims[i].stride_c);
    if (abs_stride < first_abs_stride) {
      range += abs_stride * (dim - 1);
    } else {
      if (min_larger_abs_stride == -1 || abs_stride < min_larger_abs_stride)
        min_larger_abs_stride = abs_stride;
    }
  }
  if (range >= first_abs_stride)
    return true;  // Overlap detected.

  // full_range is range of offsets when we include the zeroth (dim, stride).
  int32 full_range = range + (first_dim - 1) * first_abs_stride;
  
  if (min_larger_abs_stride != -1 && min_larger_abs_stride <= full_range)
    return true;  // Overlap detected.

  return false;  // No overlap detected.
}


template<class Real>
inline void AddTensorTensorOrder3(const TensorOperationDims *dims,
                                  Real alpha,
                                  const Real *a_data,
                                  const Real *b_data,
                                  Real *c_data,
                                  Real beta) {
  if (AddTensorTensorOrder3Blas(dims, alpha, a_data, b_data, c_data, beta))
    return;
  
  // If c_data can be accessed in multiple distinct ways with different indexes,
  // we need to scale c_data first, and set beta to 1, or the result from the
  // code below may not be correct.
  if (beta != 1.0 &&
      HasOverlapOrder2(dims[0].dim, dims[0].stride_c,
                       dims[1].dim, dims[1].stride_c)) {
    ScaleTensor(2, dims, beta, c_data);
    beta = 1.0;
  }

  // Just loop over the leading dimension.
  for (int32 i = 0; i < dims[0].dim; i++,
           a_data += dims[0].stride_a,
           b_data += dims[0].stride_b,
           c_data += dims[0].stride_c) {
    AddTensorTensorOrder2(dims + 1,
                          alpha, a_data, b_data, c_data, beta);
    // If we're going to loop multiple times and see the same data
    // for c repeatedly, we don't want to keep applying "beta".
    if (dims[0].stride_c == 0)
      beta = 1.0;
  }
  return;
}


// This function returns true if the dims/strides have non-trivial aliasing
// (that is, ignoring dims with zero stride), but it is "destructive" (will
// change that vector).  May not be used right now.
// TODO: check this.
bool HasAliasing(std::vector<std::pair<int32, int32> > *dims_strides) {
  // use the absolute values of the strides.
  for (int32 i = 0; i < dims_strides->size(); i++)
    (*dims_strides)[i].second = std::abs((*dims_strides)[i].second);
  
  // Sort from smallest to largest stride
  StrideLess sl;
  std::sort(dims_strides->begin(), dims_strides->end(), sl);
  
  for (size_t i = 0; i < dims_strides->size(); i++) {
    int32 dim = (*dims_strides)[i].first,
        stride = (*dims_strides)[i].second;
    if (dim != 1 && stride == 0)
      return true;  // Aliasing detected.
    if (i > 0 && dim != 1) {
      int32 prev_dim = (*dims_strides)[i-1].first,
          prev_stride = (*dims_strides)[i-1].second;
      if (stride < prev_stride * prev_dim)
        return true;  // aliasing detected
    }
  }
  return false;
}  


template<class Real>
void ScaleTensor(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,
                 Real alpha,
                 Real *data) {
  if (num_indexes > 1) {
    int32 dim = dims_strides[0].first, stride = dims_strides[0].second;
    for (int32 i = 0; i < dim; i++, data += stride)
      ScaleTensor(num_indexes - 1, dims_strides + 1, alpha, data);
  } else  if (num_indexes == 1) {
    int32 dim = dims_strides[0].first, stride = dims_strides[0].second;    
    cblas_Xscal(dim, alpha, data, stride);
  } else {
    KALDI_ASSERT(num_indexes == 0); // It should be rare that you reach here.
    *data *= alpha;
  }
}
// Instantiate the template above.
template
void ScaleTensor(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,
                 float alpha,
                 float *data);
template
void ScaleTensor(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,
                 double alpha,
                 double *data);


// See declaration in header for more information.
template<class Real>
void ScaleTensor(int32 num_indexes,
                 const TensorOperationDims *dims,
                 Real alpha,
                 Real *c_data) {
  if (alpha == 1.0) return;
  std::vector<std::pair<int32,int32> > dims_strides;
  dims_strides.reserve(num_indexes);
  for (int32 i = 0; i < num_indexes; i++)
    if (dims[i].stride_c != 0)
      dims_strides.push_back(std::pair<int32,int32>(dims[i].dim,
                                                    dims[i].stride_c));

  Tensor<Real> tensor(dims_strides, c_data);
  tensor.Scale(alpha);
}

// Instantiate the template above.
template
void ScaleTensor(int32 num_indexes,
                 const TensorOperationDims *dims,
                 float alpha,
                 float *c_data);
template
void ScaleTensor(int32 num_indexes,
                 const TensorOperationDims *dims,
                 double alpha,
                 double *c_data);

template<class Real>
void ApplyPowTensor(int32 num_indexes,
                    const std::pair<int32, int32> *dims_strides,
                    Real power,
                    Real *data) {
  if (num_indexes > 1) {
    int32 dim = dims_strides[0].first, stride = dims_strides[0].second;
    for (int32 i = 0; i < dim; i++, data += stride)
      ApplyPowTensor(num_indexes - 1, dims_strides + 1, power, data);
  } else if (num_indexes == 1) {
    int32 dim = dims_strides[0].first, stride = dims_strides[0].second;
    Xpow(dim, power, data, stride);
  } else {
    Xpow(1, power, data, 0);
  }
}

template 
void ApplyPowTensor(int32 num_indexes,
                    const std::pair<int32, int32> *dims_strides, 
                    float power,
                    float *data);
template 
void ApplyPowTensor(int32 num_indexes,
                    const std::pair<int32, int32> *dims_strides,   
                    double power,
                    double *data);

// Find the min value of a tensor
template<class Real>
Real MinInternal(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,
                 Real *data) {
  int32 dim = dims_strides[0].first, stride = dims_strides[0].second;
  Real min_value = std::numeric_limits<Real>::infinity(); 
  if (num_indexes > 1) {
    for (int32 i = 0; i < dim; i++, data += stride) {
      MinInternal(num_indexes - 1, dims_strides + 1, data);
    }
  } else if (num_indexes == 1) {                 
    Real tmp = Xmin(dim, data, stride);  
    min_value = std::min(min_value, tmp);
  } else {
    min_value = *data;
  }
  return min_value;
}
// Instantiate the template above
template
float MinInternal(int32 num_indexes,  
                  const std::pair<int32, int32> *dims_strides, 
                  float *data);
template
double MinInternal(int32 num_indexes,
                   const std::pair<int32, int32> *dims_strides,   
                   double *data);
//
// Find the max value of a tensor
template<class Real>
Real MaxInternal(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,
                 Real *data) {
  int32 dim = dims_strides[0].first, stride = dims_strides[0].second;
  Real max_value = -std::numeric_limits<Real>::infinity();
  if (num_indexes > 1) {
    for (int32 i = 0; i < dim; i++, data += stride) {
      MaxInternal(num_indexes - 1, dims_strides + 1, data);
    } 
  } else if (num_indexes == 1) {
    Real tmp = Xmax(dim, data, stride);
    max_value = std::max(tmp, max_value);
  } else {
    max_value = *data;
  }
  return max_value;
}
// Instantiate the template above
template
float MaxInternal(int32 num_indexes,  
                 const std::pair<int32, int32> *dims_strides, 
                 float *data);
template
double MaxInternal(int32 num_indexes,
                 const std::pair<int32, int32> *dims_strides,   
                 double *data);

//
template<class Real>
void AddTensorTensorGeneric(size_t num_indexes,
                            TensorOperationDims *dims,
                            Real alpha,
                            const Real *a_data,
                            const Real *b_data,
                            Real *c_data,
                            Real beta) {

  switch(num_indexes) {
    case 0:
      // we should not really get here unless the user does something strange.
      *c_data = beta * *c_data + alpha * *a_data * *b_data;
      return;
    case 1:
      AddTensorTensorOrder1(dims, alpha, a_data, b_data, c_data, beta);
      return;
    case 2:
      AddTensorTensorOrder2(dims, alpha, a_data, b_data, c_data, beta);
      return;
    case 3:
      AddTensorTensorOrder3(dims, alpha, a_data, b_data, c_data, beta);
      return;
    default:
      if (beta != 1.0 && FirstDimOverlaps(num_indexes, dims)) {
        ScaleTensor(num_indexes, dims, beta, c_data);
        beta = 1.0;
      }
      
      // Just loop over the leading dimension.
      for (int32 i = 0; i < dims[0].dim; i++,
               a_data += dims[0].stride_a,
               b_data += dims[0].stride_b,
               c_data += dims[0].stride_c) {
        AddTensorTensorGeneric(num_indexes - 1, dims + 1,
                               alpha, a_data, b_data, c_data, beta);
        // If we're going to loop multiple times and see the same data
        // for c repeatedly, we don't want to keep applying "beta".
        if (dims[0].stride_c == 0)
          beta = 1.0;
      }
      return;
  }
}    



template<class Real>
void AddTensorTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                             Real alpha,
                             const Real *a_data,
                             const Real *b_data,
                             Real *c_data,
                             Real beta) {
  
  std::vector<TensorOperationDims> dims(dims_in);

  ExciseUnitDims(&dims);
  ReorderSumToEnd(&dims);  // Put any summed-over dimensions at the end.
  
  size_t num_indexes = dims.size();
  AddTensorTensorGeneric(num_indexes,
                         num_indexes > 0 ? &(dims[0]) : NULL,
                         alpha, a_data, b_data, c_data, beta);  
}

// Instantiate the template above for float and double.
template
void AddTensorTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                             float alpha,
                             const float *a_data,
                             const float *b_data,
                             float *c_data,
                             float beta);
template
void AddTensorTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                             double alpha,
                             const double *a_data,
                             const double *b_data,
                             double *c_data,
                             double beta);

template<class Real>
inline void AddTensorOrder1(const TensorOperationDims *dims,
                            Real alpha,
                            const Real *a_data,
                            Real *c_data) {
  if (dims[0].stride_a != 0 && dims[0].stride_c != 0) {
    cblas_Xaxpy(dims[0].dim, alpha, a_data, dims[0].stride_a,
                c_data, dims[0].stride_c);
  } else if (dims[0].stride_a == 0 && dims[0].stride_c != 0) {
    size_t n = dims[0].dim, stride_c = dims[0].stride_c;
    Real a = alpha * a_data[0];
    for (size_t i = 0; i < n; i++)
      c_data[i * stride_c] += a;
  } else if (dims[0].stride_a != 0 && dims[0].stride_c == 0) {
    *c_data += alpha * Xsum(dims[0].dim, a_data,
                           dims[0].stride_a);
  } else {
    KALDI_ERR << "Invalid dimensions"; // both strides zero, does not make
                                       // sense.
  }
}

template<class Real>
void AddTensorGeneric(size_t order,
                      TensorOperationDims *dims,
                      Real alpha,
                      const Real *a_data,
                      Real *c_data) {
  switch(order) {
    case 0: // unlikely to reach here.
      *c_data += alpha * *a_data;
      return;      
    case 1:
      AddTensorOrder1(dims, alpha, a_data, c_data);
      return;
    default: // recurse.
      for (int32 i = 0; i < dims[0].dim; i++,
               a_data += dims[0].stride_a,
               c_data += dims[0].stride_c) {
        AddTensorGeneric(order - 1, dims + 1, alpha, a_data, c_data);
      }
  }
}



template<class Real>
void AddTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                       Real alpha,
                       const Real *a_data,
                       Real *c_data) {
  std::vector<TensorOperationDims> dims(dims_in);

  ExciseUnitDims(&dims);
  ReorderSumToEnd(&dims);  // Put any summed-over dimensions at the end.

  size_t order = dims.size();
  AddTensorGeneric(order,
                   order > 0 ? &(dims[0]) : NULL,
                   alpha, a_data, c_data);
}

// Instantiate the template
template
void AddTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                       float alpha,
                       const float *a_data,
                       float *c_data);
template
void AddTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                       double alpha,
                       const double *a_data,
                       double *c_data);


// function operator that returns true if the sum of abs(stride_a) and
// abs(stride_b) is greater.  If used as the < operation in sorting, this will
// put the smaller strides at the end, which tends to improve memory locality of
// inner loops.
struct SumStrideAbGreater {
  inline bool operator () (const TensorOperationDims &d1,
                           const TensorOperationDims &d2) {
    return (std::abs(d1.stride_a) + std::abs(d1.stride_b) >
            std::abs(d2.stride_a) + std::abs(d2.stride_b));
  }
};


/// Internal CopyTensor function, used inside CopyTensorToplevel.
/// Requires num_indexes >= 1.
template<class Real>
void CopyTensor(int32 num_indexes,
                const TensorOperationDims *dims_in,
                const Real *a_data,
                Real *b_data) {
  if (num_indexes == 1) {
    if (dims_in[0].stride_a > 0 && dims_in[0].stride_b > 0) {
      cblas_Xcopy(dims_in[0].dim, a_data, dims_in[0].stride_a,
                  b_data, dims_in[0].stride_b);
    } else {
      int32 dim = dims_in[0].dim, stride_a = dims_in[0].stride_a,
          stride_b = dims_in[0].stride_b;
      if (stride_a == 0) {
        Real a = *a_data;
        for (int32 i = 0; i < dim; i++)
          b_data[i * stride_b] = a;
      } else {
        // we'd only be here in weird cases like negative strides.
        // Note: stride_b == 0 is not going to show up, it's not
        // allowed because it doesn't make sense, and we checked
        // at a higher level.
        for (int32 i = 0; i < dim; i++)
          b_data[i * stride_b] = a_data[i * stride_a];
      }
    }
  } else {
    KALDI_ASSERT(num_indexes > 1);
    int32 dim = dims_in[0].dim, stride_a = dims_in[0].stride_a,
        stride_b = dims_in[0].stride_b;
    for (int32 i = 0; i < dim; i++, a_data += stride_a, b_data += stride_b)
      CopyTensor(num_indexes - 1, dims_in + 1, a_data, b_data);
  }  
}

template<class Real>
void CopyTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                        const Real *a_data,
                        Real *b_data) {
  std::vector<TensorOperationDims> dims(dims_in);
  ExciseUnitDims(&dims);

  // Sort so the dimensions with the smallest abs(stride_a) + abs(stride_b) are
  // last.
  SumStrideAbGreater sg;
  std::sort(dims.begin(), dims.end(), sg);

  while (dims.size() > 1) {
    // Try to combine the last two index-positions... this is applicable in
    // cases like where there is no gap between the rows of a matrix laid out in
    /// memory.
    TensorOperationDims &last = dims[dims.size() - 1],
        &next_to_last = dims[dims.size() - 2];
    if (next_to_last.stride_a == last.stride_a * last.dim &&
        next_to_last.stride_b == last.stride_b * last.dim) {
      next_to_last.dim *= last.dim;
      next_to_last.stride_a = last.stride_a;
      next_to_last.stride_b = last.stride_b;
      dims.pop_back();
    } else {
      break;
    }
  }
  if (dims.size() == 0)
    *b_data = *a_data;
  else
    CopyTensor(dims.size(), &(dims[0]), a_data, b_data);
}

// Instantiate the template above.
template
void CopyTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                        const float *a_data,
                        float *b_data);
template
void CopyTensorToplevel(const std::vector<TensorOperationDims> &dims_in,
                        const double *a_data,
                        double *b_data);

} // namespace kaldi
