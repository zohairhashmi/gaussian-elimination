/* 
Gaussian elimination using Message Passing Interface (MPI)
*/

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include <mpi.h>

/* Program Parameters */
#define MAXN 5000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) { // argc is the number of arguments, argv is the array of arguments e.g. 
  // argc : N. agrv : random seed
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  // /* Start Clock */
  // printf("\nStarting clock.\n");
  // gettimeofday(&etstart, &tzdummy);
  // etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  // /* Stop Clock */
  // gettimeofday(&etstop, &tzdummy);
  // etstop2 = times(&cputstop);
  // printf("Stopped clock.\n");
  // usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  // usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  // /* Display timing results */
  // printf("\nElapsed time = %g ms.\n",
	//  (float)(usecstop - usecstart)/(float)1000);

  // printf("(CPU times are accurate to the nearest %g ms)\n",
	//  1.0/(float)CLOCKS_PER_SEC * 1000.0);
  // printf("My total CPU time for parent = %g ms.\n",
	//  (float)( (cputstop.tms_utime + cputstop.tms_stime) -
	// 	  (cputstart.tms_utime + cputstart.tms_stime) ) /
	//  (float)CLOCKS_PER_SEC * 1000);
  // printf("My system CPU time for parent = %g ms.\n",
	//  (float)(cputstop.tms_stime - cputstart.tms_stime) /
	//  (float)CLOCKS_PER_SEC * 1000);
  // printf("My total CPU time for child processes = %g ms.\n",
	//  (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
	// 	  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	//  (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss() {
  // Intialize MPI
  MPI_Init(NULL, NULL);

  // Start the timer
  double start_time = MPI_Wtime();

  // Get the total number of processes
  int numtasks;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;

  // Calculate number of rows per process
  int rows_per_process = N / numtasks;
  int remainder = N % numtasks;

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create chunks of data to send to each process
  float *local_A = (float *)malloc(rows_per_process * N * sizeof(float));
  float *local_B = (float *)malloc(rows_per_process * sizeof(float));

  // Scatter the matrix A and vector B across all processes
  MPI_Scatter(A, rows_per_process * N, MPI_FLOAT, local_A, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatter(B, rows_per_process, MPI_FLOAT, local_B, rows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Perform Gaussian elimination
  for (norm = 0; norm < N - 1; norm++) {
    // Broadcast the normalization row to all processes
    MPI_Bcast(A[norm], N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each process works on its assigned rows
    for (row = 0; row < rows_per_process; row++) {
      if (rank * rows_per_process + row != norm) { // Skip the normalization row
        multiplier = local_A[row * N + norm] / A[norm][norm];
        for (col = norm; col < N; col++) {
          local_A[row * N + col] -= A[norm][col] * multiplier;
        }
        local_B[row] -= B[norm] * multiplier;
      }
    }

    // Gather the updated local_A and local_B arrays back to the root process
    MPI_Gather(local_A, rows_per_process * N, MPI_FLOAT, A, rows_per_process * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_B, rows_per_process, MPI_FLOAT, B, rows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  // Perform back substitution
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N - 1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
  
  // Stop the timer
  double end_time = MPI_Wtime();

  // Calculate the elapsed time
  double elapsed_time = end_time - start_time;

  // Print the elapsed time only by the master process
  if (rank == 0) {
    printf("Elapsed time: %f seconds\n", elapsed_time);
  }

  // Free allocated memory
  free(local_A);
  free(local_B);

  // Finalize MPI
  MPI_Finalize();
}
