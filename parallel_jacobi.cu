//
// Implementation of the iterative Jacobi method.
//
// Given a known, diagonally dominant matrix A and a known vector b, we aim to
// to find the vector x that satisfies the following equation:
//
//     Ax = b
//
// We first split the matrix A into the diagonal D and the remainder R:
//
//     (D + R)x = b
//
// We then rearrange to form an iterative solution:
//
//     x' = (b - Rx) / D
//
// More information:
// -> https://en.wikipedia.org/wiki/Jacobi_method
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

static int N;
static int MAX_ITERATIONS;
static int SEED;
static float CONVERGENCE_THRESHOLD;

#define BLOCK_SIZE 256

#define SEPARATOR "------------------------------------\n"

float get_timestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void total(float * input, float * output, int len) {
	__shared__ float partialSum[2 * BLOCK_SIZE];
	
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	
	partialSum[t] = (t < len) ? input[start + t] : 0;
	partialSum[blockDim.x + t] = ((blockDim.x + t) < len) ? input[start + blockDim.x + t] : 0;
	
	//@@ Load a segment of the input vector into shared memory
	
	for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1) {
		__syncthreads();
		if(t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	
	if(t == 0) {
		output[blockIdx.x + t] = partialSum[t];
	}
  __syncthreads();
}
void zeros(float *host, int n){
  #pragma omp parallel for
  for (int i = 0; i < n; i++){
    host[i] = 0;
  }
  #pragma omp barrier
}
// Run the Jacobi solver
// Returns the number of iterations performed
int run(float *A, float *b, float *x, float *xtmp)
{
  int itr;
  int row, col;
  float dot;
  float diff;
  float sqdiff;
  float *ptrtmp;
  float *hostInput = (float*)malloc(N*sizeof(float));
  float * hostOutput; // The output list
  float * deviceInput;
  float * deviceOutput;
  int numInputElements = N; // number of elements in the input list
  int numOutputElements = numInputElements / (BLOCK_SIZE<<1);

  if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
  hostOutput = (float*) malloc(numOutputElements * sizeof(float));
  cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
	cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));
	
  // Loop until converged or maximum iterations reached
  itr = 0;
  do
  {
    // Perfom Jacobi iteration
    for (row = 0; row < N; row++)
    {
      dot = 0.0;
      zeros(hostInput, N-1);
      #pragma omp parallel for
      for (col = 0; col < N; col++)
      {
        if (row != col)
          hostInput[col] = A[row + col * N] * x[col];
          //printf("lo que se le esta %f\n", hostInput[col]);}
      }
      #pragma omp barrier
      /**/

      cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);
      dim3 DimGrid((numInputElements - 1)/BLOCK_SIZE + 1, 1, 1);
	    dim3 DimBlock(BLOCK_SIZE, 1, 1);

      //@@ Launch the GPU Kernel here
	    total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
	
      cudaDeviceSynchronize();

      //@@ Copy the GPU memory back to the CPU here
	    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
      //for (int k = 0; k < numOutputElements; k++){
        //printf("%f, ", hostOutput[k]);
      //}
      //printf("\n");
      for (int ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
      }

      dot = hostOutput[0];
      //printf("%f\n", dot);

      /**/
      xtmp[row] = (b[row] - dot) / A[row + row * N];
    }


    // Swap pointers
    ptrtmp = x;
    x = xtmp;
    xtmp = ptrtmp;

    // Check for convergence
    sqdiff = 0.0;
    for (row = 0; row < N; row++)
    {
      diff = xtmp[row] - x[row];
      // printf("soy diff %f\n", diff);
      sqdiff += diff * diff;
    }

    itr++;
  } while ((itr < MAX_ITERATIONS) && (sqrt(sqdiff) > CONVERGENCE_THRESHOLD));
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  
  free(hostInput);
  free(hostOutput);
  return sqdiff;
}

float execute()
{

  float *A = (float*)malloc(N * N * sizeof(float));
  float *b = (float*)malloc(N * sizeof(float));
  float *x = (float*)malloc(N * sizeof(float));
  float *xtmp = (float*)malloc(N * sizeof(float));

  printf(SEPARATOR);
  printf("Matrix size:            %dx%d\n", N, N);

  float total_start = get_timestamp();

  // Initialize data
  srand(SEED);
  for (int row = 0; row < N; row++)
  {
    float rowsum = 0.0;
    for (int col = 0; col < N; col++)
    {
      float value = rand() / (float)RAND_MAX;
      A[row + col * N] = value;
      rowsum += value;
    }
    A[row + row * N] += rowsum;
    b[row] = rand() / (float)RAND_MAX;
    x[row] = 0.0;
  }

  // Run Jacobi solver
  float solve_start = get_timestamp();
  int itr = run(A, b, x, xtmp);
  float solve_end = get_timestamp();

  printf("Solver runtime = %lf seconds\n", (solve_end - solve_start));

  free(A);
  free(b);
  free(x);
  free(xtmp);
  return (solve_end - solve_start);
}

int main()
{
  MAX_ITERATIONS = 200;
  CONVERGENCE_THRESHOLD = 0.001;
  SEED = 0;
  int mul = 200;
  int iter = 10;
  float history[iter];
  int i;
  for (i = 1; i <= iter; i++)
  {
    N = i* mul;
    history[i - 1] = execute();
  }
  for (i = 0; i < iter; i++)
  {
    printf("%f;", history[i]);
  }
  return 0;
}