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

static int N;
static int MAX_ITERATIONS;
static int SEED;
static double CONVERGENCE_THRESHOLD;

#define SEPARATOR "------------------------------------\n"

double get_timestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Run the Jacobi solver
// Returns the number of iterations performed
int run(double *A, double *b, double *x, double *xtmp)
{
  int itr;
  int row, col;
  double dot;
  double diff;
  double sqdiff;
  double *ptrtmp;

  // Loop until converged or maximum iterations reached
  itr = 0;
  do
  {
    // Perfom Jacobi iteration
    for (row = 0; row < N; row++)
    {
      dot = 0.0;
      for (col = 0; col < N; col++)
      {
        if (row != col)
          dot += A[row + col * N] * x[col];
      }
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
      //printf("soy diff %f\n", diff);
      sqdiff += diff * diff;
    }

    itr++;
  } while ((itr < MAX_ITERATIONS) && (sqrt(sqdiff) > CONVERGENCE_THRESHOLD));

  return sqdiff;
}

double execute()
{

  double *A = malloc(N * N * sizeof(double));
  double *b = malloc(N * sizeof(double));
  double *x = malloc(N * sizeof(double));
  double *xtmp = malloc(N * sizeof(double));

  printf(SEPARATOR);
  printf("Matrix size:            %dx%d\n", N, N);

  double total_start = get_timestamp();

  // Initialize data
  srand(SEED);
  for (int row = 0; row < N; row++)
  {
    double rowsum = 0.0;
    for (int col = 0; col < N; col++)
    {
      double value = rand() / (double)RAND_MAX;
      A[row + col * N] = value;
      rowsum += value;
    }
    A[row + row * N] += rowsum;
    b[row] = rand() / (double)RAND_MAX;
    x[row] = 0.0;
  }

  // Run Jacobi solver
  double solve_start = get_timestamp();
  int itr = run(A, b, x, xtmp);
  double solve_end = get_timestamp();

  printf("Solver runtime = %lf seconds\n", (solve_end - solve_start));

  free(A);
  free(b);
  free(x);
  free(xtmp);
  return itr;
}

int main()
{
  MAX_ITERATIONS = 200;
  CONVERGENCE_THRESHOLD = 0.001;
  SEED = 0;
  int mul = 200;
  int iter = 10;
  double history[iter];
  int i;
  for (i = 1; i <= iter; i++)
  {
    N = i * mul;
    history[i - 1] = execute();
  }
  for (i = 0; i < iter; i++)
  {
    printf("%f;", history[i]);
  }
  return 0;
}