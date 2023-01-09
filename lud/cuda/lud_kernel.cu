#include <cuda.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>


#include "common.h"

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

void lud_cuda(float *m, int matrix_dim, cublasHandle_t handle)
{

  int* pivots = (int*) malloc(matrix_dim * sizeof(int));
  int* info = (int*) malloc(sizeof(int));
   double * const * p = (double * const *)&m;

  cublasDgetrfBatched(handle, matrix_dim, p, matrix_dim, pivots, info, 1);

  free(pivots);
  free(info);
}