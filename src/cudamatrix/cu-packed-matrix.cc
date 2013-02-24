// cudamatrix/cu-packed-matrix.cc

// Copyright 2009-2013  Johns Hopkins University (author: Daniel Povey)
//                      Karel Vesely

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


#if HAVE_CUDA==1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-math.h"
#include "cu-packed-matrix.h"

namespace kaldi {

template<typename Real>
void CuPackedMatrix<Real>::Resize(MatrixIndexT rows,
                                  MatrixResizeType resize_type) {
  // This code does not currently support the other resize_type options.
  KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);

  if (this->num_rows_ == rows) {
    if (resize_type == kSetZero) this->SetZero();
    return;
  }

  if (this->num_rows_ != 0)
    this->Destroy();
  if (rows == 0) return;  
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&this->data_), num_bytes));

    this->num_rows_ = rows;
    if (resize_type == kSetZero) this->SetZero();
  } else
#endif
  { // Let the initializer of SpMatrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
    SpMatrix<Real> mat(rows, resize_type);
    this->Swap(&mat);
  }
}


template<typename Real>
void CuPackedMatrix<Real>::Destroy() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    if (this->data_ != NULL) {
      CU_SAFE_CALL(cudaFree(this->data_));
    }
  } else
  #endif
  {
    if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
  }
  this->data_ = NULL;
  this->num_rows_ = 0;
}

template<typename Real>
void CuPackedMatrix<Real>::Swap(PackedMatrix<Real> *mat) {
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
        // *this is empty, but mat is nonempty.
        Resize(mat->num_rows_, kUndefined);
        CopyFromPacked(*mat);
        mat->Resize(0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (mat->num_rows_ != 0) {
        // Both *this and *mat are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        Matrix<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        mat->Swap(&temp); // now mat has data from *this, temp has
        // data from mat.
        this->Swap(mat); // copy data in mat to *this, which is now empty.
      } else { // *this is full but *mat is empty.
        mat->Resize(this->num_rows_, kUndefined);
        this->CopyToMat(mat);
        this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(mat->data_, this->data_);
    std::swap(mat->num_rows_, this->num_rows_);
  }
}

template<typename Real>
void CuPackedMatrix<Real>::Swap(Matrix<Real> *mat) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
	// *this is empty, but mat is nonempty.
	Resize(mat->num_rows_, kUndefined);
	CopyFromMat(*mat);
	mat->Resize(0, 0);
      }
      // else both are empty
    } else {
      if (mat->num_rows_ != 0) {
	// Both *this and *mat are nonempty.  Recurse to simpler cases.
	// this could be done more efficiently in the case where
	// the size does not change.
	Matrix<Real> temp;
	this->Swap(&temp); // now temp is full, *this is empty.
	mat->Swap(&temp); // now mat has data from *this, temp has 
	// data from mat.
	this->Swap(mat); // copy data in mat to *this, which is now empty.
      } else { // *this is full but *mat is empty.
	mat->Resize(this->num_rows_, this->num_cols_, kUndefined);
	this->CopyToMat(mat);
	this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(mat->data_, this->data_);
    std::swap(mat->num_cols_, this->num_cols_);
    std::swap(mat->num_rows_, this->num_rows_);
    std::swap(mat->stride_, this->stride_);
  }
}

template<typename Real>
void CuPackedMatrix<Real>::CopyFromPacked(const CuPackedMatrix<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_);
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy(data_, src.data_, num_bytes,
                            cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromPacked1",tim.Elapsed());
  } else
#endif
  {
    //Mat().CopyFromPacked(src.Mat());
    memcpy(data_, src.Data(), SizeInBytes());
  }
}



template<typename Real>
void CuPackedMatrix<Real>::CopyFromPacked(const PackedMatrix<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_);
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);
    MatrixIndexT width = src.NumCols() * sizeof(Real);
    MatrixIndexT dst_pitch = stride_ * sizeof(Real);
    //MatrixIndexT src_pitch = src.Stride() * sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.data_, num_bytes,
			      width, src.NumRows(), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromPacked2",tim.Elapsed());
  } else
#endif
  {
    //Mat().CopyFromPacked(src);
    memcpy(data_, src.Data(), SizeInBytes());
  }
}

template<typename Real>
void CuPackedMatrix<Real>::CopyFromMat(const Matrix<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    
    MatrixIndexT dst_pitch = stride_ * sizeof(Real);
    MatrixIndexT src_pitch = src.Stride() * sizeof(Real);
    MatrixIndexT width = src.NumCols() * sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.data_, src_pitch,
			      width, src.num_rows_, cudaMemcpyDeviceToDevice));
    
    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
  } else
#endif
    {
      Mat().CopyFromMat(src);
      //Mat().CopyFromMat(src.Mat());
    }
}


template<typename Real>
void CuPackedMatrix<Real>::CopyToMat(PackedMatrix<Real> *dst) const {
  KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
  
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 

    Timer tim;

    MatrixIndexT src_pitch = stride_*sizeof(Real);
    MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
    MatrixIndexT width = NumCols()*sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(dst->data_, dst_pitch, this->data_, src_pitch,
                            width, this->num_rows_, cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
  #endif
  {
    memcpy(data_, dst->Data(), SizeInBytes());
    //dst->CopyFromPacked(Mat());
  }
}

template<typename Real>
void CuPackedMatrix<Real>::CopyToMat(Matrix<Real> *dst) const {
  KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
  
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    
    Timer tim;
    
    MatrixIndexT src_pitch = stride_*sizeof(Real);
    MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
    MatrixIndexT width = NumCols()*sizeof(Real);
    CU_SAFE_CALL(cudaMemcpy2D(dst->data_, dst_pitch, this->data_, src_pitch,
			      width, this->num_rows_, cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
#endif
    {
      memcpy(data_, dst->Data(), SizeInBytes());
      //dst->CopyFromPacked(Mat());                                                   
    }
}


/*
template<typename Real>
void CuPackedMatrix<Real>::CopyRowsFromPacked(int32 r, const CuPackedMatrix<Real> &src, int32 src_ro, int32 dst_ro) {
  KALDI_ASSERT(r+src_ro <= src.NumRows());
  KALDI_ASSERT(r+dst_ro <= NumRows());
  KALDI_ASSERT(NumCols() == src.NumCols());
   
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);

    const Real *p_src = src.Data() + src_ro*src.Stride();  
    Real *p_dst = data_ + dst_ro*stride_;

    CU_SAFE_CALL(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, r, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(Data()+dst_ro*stride_, src.Data()+src_ro*src.Stride(), r*stride_*sizeof(Real));
  }
} */



template<typename Real>
void CuMatrix<Real>::Read(std::istream &is, bool binary) {
  Matrix<Real> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}

template<typename Real>
void CuMatrix<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<Real> temp(this->num_rows_, this->num_cols_, kUndefined);
  this->CopyToMat(&temp);
  temp.Write(os, binary); 
}

template<typename Real>
void CuPackedMatrix<Real>::SetZero() {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemset(data_, 0, num_rows_*stride_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero", tim.Elapsed());
  } else
  #endif
  {
    Mat().SetZero();
  }
}



/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrix<Real> &mat) {
  Matrix<Real> temp;
  mat.CopyToMat(&temp);
  out << temp;
  return out;
}



/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real> 
void CuPackedMatrix<Real>::Set(Real value) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Set(value);
  }
}



template<typename Real> 
void CuPackedMatrix<Real>::Add(Real value) { 
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Add(value);
  }
}


template<typename Real> 
void CuPackedMatrix<Real>::Scale(Real value) { 
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_scale(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(value);
  }
}



template<typename Real> 
void CuPackedMatrix<Real>::ApplyLog() { 
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_apply_log(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().ApplyLog();
  }
}



template<typename Real>
void CuPackedMatrix<Real>::MulElements(const CuPackedMatrix<Real>& A) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    //KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());
    KALDI_ASSERT(stride_ == A.Stride());
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_elements(dimGrid, dimBlock, data_, A.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().MulElements(A.Mat());
  }
}



template<typename Real>
void CuPackedMatrix<Real>::MulColsVec(const CuVectorBase<Real> &scale) {
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_cols_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().MulColsVec(scale.Vec());
  }
}



template<typename Real>
void CuPackedMatrix<Real>::MulRowsVec(const CuVectorBase<Real> &scale) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Mat().MulRowsVec(scale.Vec());
  }
}



template<typename Real>
void CuPackedMatrix<Real>::DivRowsVec(const CuVectorBase<Real> &div) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(div.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_div_rows_vec(dimGrid, dimBlock, data_, div.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
#endif
  {
    Vector<Real> temp(div.Vec()); // will copy.
    temp.InvertElements();
    Mat().MulRowsVec(temp);
  }
}



template<typename Real>
void CuPackedMatrix<Real>::AddMat(Real alpha, const CuPackedMatrix<Real>& A, Real beta) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(A.NumRows() == NumRows());
    KALDI_ASSERT(A.NumCols() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_mat(dimGrid, dimBlock, alpha, A.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(beta);
    Mat().AddMat(alpha, A.Mat());
  }
}



template<typename Real>
void CuPackedMatrix<Real>::AddVecToCols(Real alpha,
                                      const CuVectorBase<Real> &col,
                                      Real beta) { 
  if (col.Dim() != NumRows()) {
    KALDI_ERR << "Non matching dimensions: Rows:" << NumRows() << " VectorDim:" << col.Dim();
  }

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_cols(dimGrid, dimBlock, alpha, col.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToCols(alpha, col.Vec());
  }
}


template<typename Real>
void CuPackedMatrix<Real>::AddVecToRows(Real alpha,
                                        const CuVectorBase<Real> &row,
                                        Real beta) { 
  if (row.Dim() != NumCols()) {
    KALDI_ERR << "Non matching dimensions: Cols:" << NumCols() << " VectorDim:" << row.Dim();
  }
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToRows(alpha, row.Vec());
  }
}





// Instantiate class CuPackedMatrix for float and double.
template class CuPackedMatrix<float>;
template class CuPackedMatrix<double>;


} // namespace kaldi
