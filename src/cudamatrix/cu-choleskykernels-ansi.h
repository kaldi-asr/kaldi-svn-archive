#ifndef KALDI_CUDAMATRIX_CU_CHOLESKYKERNELS_ANSI_H_
#define KALDI_CUDAMATRIX_CU_CHOLESKYKERNELS_ANSI_H_

#include <stdlib.h>
#include <stdio.h>

#include "cudamatrix/cu-matrixdim.h"

#if HAVE_CUDA==1

extern "C" {

/*********************************************************
 * float CUDA kernel calls
 */
void cudaF_factorize_diagonal_block(float* A, int block_offset, MatrixDim d);
void cudaF_strip_update(float* A, int block_offset, int n_rows_padded, int n_remaining_blocks);
void cudaF_diag_update(float* A, int block_offset, int n_rows_padded, int n_remaining_blocks);
void cudaF_lo_update(float* A, int block_offset, int n_blocks, int n_rows_padded, int n_remaining_blocks);


/*********************************************************
 * double CUDA kernel calls
 */
void cudaD_factorize_diagonal_block(double* A, int block_offset, MatrixDim d);
void cudaD_strip_update(double* A, int block_offset, int n_rows_padded, int n_remaining_blocks);
void cudaD_diag_update(double* A, int block_offset, int n_rows_padded, int n_remaining_blocks);
void cudaD_lo_update(double* A, int block_offset, int n_blocks, int n_rows_padded, int n_remaining_blocks);
}

#endif // HAVE_CUDA

#endif
