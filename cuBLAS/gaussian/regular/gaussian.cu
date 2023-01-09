/*-----------------------------------------------------------
 **   gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The Cublas library was used to 
 **   implement LU decomposition and Gaussian elimination.
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 ** Modified by Yujie Chen, 01/07/2022
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>
#include <cublas_v2.h>

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 4
#endif

#define EEXIT(fmt, ...)                             \
	do {											\
		fprintf(stderr, "L%d: " fmt "\n",           \
				__LINE__, ##__VA_ARGS__);           \
		exit(2);                                    \
	} while(0)

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

void InitProblemOnce(char *filename);
void InitPerRun();
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
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

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }
}


int main(int argc, char *argv[])
{
  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    int verbose = 1;
    int i, j;
    char flag;
    if (argc < 2) {
        printf("Usage: gaussian -f filename / -s size [-q]\n\n");
        printf("-q (quiet) suppresses printing the matrix and result values.\n");
        printf("-f (filename) path of input file\n");
        printf("-s (size) size of matrix. Create matrix and rhs in this program \n");
        printf("The first line of the file contains the dimension of the matrix, n.");
        printf("The second line of the file is a newline.\n");
        printf("The next n lines contain n tab separated values for the matrix.");
        printf("The next line of the file is a newline.\n");
        printf("The next line of the file is a 1xn vector with tab separated values.\n");
        printf("The next line of the file is a newline. (optional)\n");
        printf("The final line of the file is the pre-computed solution. (optional)\n");
        printf("Example: matrix4.txt:\n");
        printf("4\n");
        printf("\n");
        printf("-0.6	-0.5	0.7	0.3\n");
        printf("-0.3	-0.9	0.3	0.7\n");
        printf("-0.4	-0.5	-0.3	-0.8\n");	
        printf("0.0	-0.1	0.2	0.9\n");
        printf("\n");
        printf("-0.85	-0.68	0.24	-0.53\n");	
        printf("\n");
        printf("0.7	0.0	-0.4	-0.5\n");
        exit(0);
    }
    
    //PrintDeviceProperties();
    //char filename[100];
    //sprintf(filename,"matrices/matrix%d.txt",size);

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              Size = atoi(argv[i]);
	      printf("Create matrix internally in parse, size = %d \n", Size);

	      a = (float *) malloc(Size * Size * sizeof(float));
        a = (float *) malloc(Size * Size * sizeof(float));
        b = (float *) malloc(Size * sizeof(float));

	      create_matrix(a, Size);

	      for (j =0; j< Size; j++)
	    	b[j]=1.0;
              break;

            case 'f': // platform
              i++;
	      printf("Read file from %s \n", argv[i]);
	      InitProblemOnce(argv[i]);
              break;
            case 'q': // quiet
	      verbose = 0;
              break;
	  }
      }
    }
    
    cublasHandle_t handle;
    cublasStatus_t status;

    // Allocate managed memory
    float *A, *B;
    float **AP, **BP, **AP_d, **BP_d;
    int *Pivot, *Info;

    A = (float *) malloc(Size * Size * sizeof(float));
    B = (float *) malloc(Size * sizeof(float));

    AP = (float**) malloc(Size * sizeof(float*));
    for (int i = 0; i < Size; i++) {
      AP[i] = (float*) malloc(Size * sizeof(float));
    }
    BP = (float**) malloc(Size * sizeof(float*));
    for (int i = 0; i < Size; i++) {
      BP[i] = (float*) malloc(sizeof(float));
    }

    Info = (int *) malloc(Size * sizeof(int));
    Pivot = (int *) malloc(sizeof(int));

    //Copy memory to double float pointer A and B
    memcpy(A, a, Size*Size*sizeof(float));
      for(i=0;i<Size;i++){
        AP[i] = A+i*Size;
      }
	    for(i=0;i<Size;i++){
        BP[i] = B+i*Size;
      }

    //Allocate GPU memory
    cudaMalloc((void**)&AP_d, Size*sizeof(float*));
    for (int i = 0; i < Size; i++) {
      cudaMalloc((void**)&AP[i], Size*sizeof(float));
    }
    cudaMalloc((void**)&BP_d, Size*sizeof(float*));
    for (int i = 0; i < Size; i++) {
      cudaMalloc((void**)&BP[i], sizeof(float));
    }

    //Create a handle for cublas labrary
    cublasCreate(&handle);
  
    //begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
    
    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(AP_d, AP, Size*Size*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(BP_d, AP, Size*sizeof(float*), cudaMemcpyHostToDevice);

    // Perform LU Decomposition
    status = cublasSgetrfBatched(handle, Size, AP_d, Size, Pivot, Info, 1);

    // Perform Gaussian elimination
    cublasSgetrsBatched(handle, CUBLAS_OP_N, Size, 1, AP_d, Size, Pivot, BP_d, Size, Info, 1);
    cudaDeviceSynchronize();

    // Copy the GPU memory back to the CPU here
    cudaMemcpy(AP, AP_d, Size*Size*sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(BP, BP_d, Size*sizeof(float*), cudaMemcpyDeviceToHost);

    // End timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);

    if (status != CUBLAS_STATUS_SUCCESS)
		  EEXIT("failed on cublasSgetrfBatched");

    printf("\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-6);


    // Free the GPU memory here
    cudaFree(AP_d);
    cudaFree(BP_d);
    cudaFree(A);
    cudaFree(B);
    cudaFree(Info);
    cudaFree(Pivot);
    cudaFree(AP);
    cudaFree(BP);

    free(A);
    free(B);
    cublasDestroy(handle);
}
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties(){
	cudaDeviceProp deviceProp;  
	int nDevCount = 0;  
	
	cudaGetDeviceCount( &nDevCount );  
	printf( "Total Device found: %d", nDevCount );  
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx )  
	{  
	    memset( &deviceProp, 0, sizeof(deviceProp));  
	    if( cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx))  
	        {
				printf( "\nDevice Name \t\t - %s ", deviceProp.name );  
			    printf( "\n**************************************");  
			    printf( "\nTotal Global Memory\t\t\t - %lu KB", deviceProp.totalGlobalMem/1024 );  
			    printf( "\nShared memory available per block \t - %lu KB", deviceProp.sharedMemPerBlock/1024 );  
			    printf( "\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock );  
			    printf( "\nWarp size in threads \t\t\t - %d", deviceProp.warpSize );  
			    printf( "\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch );  
			    printf( "\nMaximum threads per block \t\t - %d", deviceProp.maxThreadsPerBlock );  
			    printf( "\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );  
			    printf( "\nMaximum Thread Dimension (grid) \t - %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );  
			    printf( "\nTotal constant memory \t\t\t - %zu bytes", deviceProp.totalConstMem );  
			    printf( "\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major, deviceProp.minor );  
			    printf( "\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate );  
			    printf( "\nTexture Alignment \t\t\t - %zu bytes", deviceProp.textureAlignment );  
			    printf( "\nDevice Overlap \t\t\t\t - %s", deviceProp. deviceOverlap?"Allowed":"Not Allowed" );  
			    printf( "\nNumber of Multi processors \t\t - %d\n\n", deviceProp.multiProcessorCount );  
			}  
	    else  
	        printf( "\n%s", cudaGetErrorString(cudaGetLastError()));  
	}  
}
 
 
/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename)
{
	//char *filename = argv[1];
	
	//printf("Enter the data file name: ");
	//scanf("%s", filename);
	//printf("The file name is: %s\n", filename);
	
	fp = fopen(filename, "r");
	
	fscanf(fp, "%d", &Size);	
	 
	a = (float *) malloc(Size * Size * sizeof(float));
	 
	InitMat(a, Size, Size);
	//printf("The input matrix a is:\n");
	//PrintMat(a, Size, Size);
	b = (float *) malloc(Size * sizeof(float));
	
	InitAry(b, Size);
	//printf("The input array b is:\n");
	//PrintAry(b, Size);
		
	 m = (float *) malloc(Size * Size * sizeof(float));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */

void InitMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+Size*i+j);
		}
	}  
}
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2f ", ary[i]);
	}
	printf("\n\n");
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

