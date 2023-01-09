#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "cusolver_utils.h"

struct timeval t_start, t_end; 
void cputimer_start(){
    gettimeofday(&t_start, 0);
}
void cputimer_stop(){
    gettimeofday(&t_end, 0);
    double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
    // printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
    printf("%.0f", time);
}


void
create_matrix(float **mp, int size){
  float *m;
  int i,j;
  float lamda = -0.001;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }

  m = (float*) malloc(sizeof(float)*size*size);
//   if ( m == NULL) {
//       return RET_FAILURE;
//   }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }

  *mp = m;

//   return RET_SUCCESS;
}


int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    const int m = atoi(argv[1]);
    const int lda = m;
    const int ldb = m;

    /*       
     *       | 1 2 3  |
     *   A = | 4 5 6  |
     *       | 7 8 10 |
     *
     * without pivoting: A = L*U
     *       | 1 0 0 |      | 1  2  3 |
     *   L = | 4 1 0 |, U = | 0 -3 -6 |
     *       | 7 2 1 |      | 0  0  1 |
     *
     * with pivoting: P*A = L*U
     *       | 0 0 1 |
     *   P = | 1 0 0 |
     *       | 0 1 0 |
     *
     *       | 1       0     0 |      | 7  8       10     |
     *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
     *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
     */
    float *mat;
    create_matrix(&mat, m);
    
    // const std::vector<double> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    // const std::vector<double> B = {1.0, 2.0, 3.0};
    // std::vector<double> A(m*m, 0.0);
    // for (int i = 0; i < m; i++) {
    //   for (int j = 0; j < m; j++) {
    //     A[i*m+j] = mat[i*m+j];
    //   }
    // }
    std::vector<double> A(mat, mat + m*m);
    const std::vector<double> B(m, 1);
    std::vector<double> X(m, 0);
    std::vector<double> LU(lda * m, 0);
    std::vector<int> Ipiv(m, 0);
    int info = 0;

    double *d_A = nullptr; /* device copy of A */
    double *d_B = nullptr; /* device copy of B */
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const int pivot_on = 0;

    // if (pivot_on) {
    //     printf("pivot is on : compute P*A = L*U \n");
    // } else {
    //     printf("pivot is off: compute A = L*U (not numerically stable)\n");
    // }

    // printf("A = (matlab base-1)\n");
    // print_matrix(m, m, A.data(), lda);
    // printf("=====\n");

    // printf("B = (matlab base-1)\n");
    // print_matrix(m, 1, B.data(), ldb);
    // printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of getrf */
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: LU factorization */
    cputimer_start();
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info));
    }
    cputimer_stop();
    if (pivot_on) {
        CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(
        cudaMemcpyAsync(LU.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 > info) {
        // printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on) {
        // printf("pivoting sequence, matlab base-1\n");
        for (int j = 0; j < m; j++) {
            printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
        }
    }
    // printf("L and U = (matlab base-1)\n");
    // print_matrix(m, m, LU.data(), lda);
    // printf("=====\n");

    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
    // if (pivot_on) {
    //     CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
    //                                     d_A, lda, d_Ipiv, d_B, ldb, d_info));
    // } else {
    //     CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
    //                                     d_A, lda, NULL, d_B, ldb, d_info));
    // }

    // CUDA_CHECK(
    //     cudaMemcpyAsync(X.data(), d_B, sizeof(double) * X.size(), cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // printf("X = (matlab base-1)\n");
    // print_matrix(m, 1, X.data(), ldb);
    // printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
