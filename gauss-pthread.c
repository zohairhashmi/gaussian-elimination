/*GAUSSIAN ELIMINATION - OPENMP VERSION
 * 
 * Zohair Hashmi - 668913771
 * 
 * In the following code, the original gauss function is 
 * modified by adding Pthreads to parallelize the code. 
 * The parallel region is defined by the gauss_thread function,
 * and the work is divided among the threads using the global_row
 * variable and the global_row_lock mutex. The barrier is used to
 * synchronize the threads.
 * 
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
#include <pthread.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
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

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */

/*Defining Global Variables*/
#define CHUNK_SIZE 1                // purpose: to divide the work into chunks that are fed to the threads
long global_row = 0;                // purpose: to keep track of the row that is being processed
int NTHREADS = 20;                  // purpose: to define the number of threads to be used
pthread_mutex_t global_row_lock;    // purpose: to lock the global_row variable
pthread_barrier_t barrier;          // purpose: to synchronize the threads

/*Computing Gauss using Parallel Pthreads*/
void *gauss_thread(void *threadid) {
  long row, col, row_max;           // row_max: to keep track of the maximum row of the chunk
  float multiplier;

  /* Gaussian elimination */
  for (long norm = 0; norm < N -1; norm++){ // iterate through the columns
    global_row = norm + 1; 

    while(global_row < N){ // iterate through the rows
      pthread_mutex_lock(&global_row_lock); // lock the global_row variable
      row = global_row; 
      global_row += CHUNK_SIZE; 
      pthread_mutex_unlock(&global_row_lock); // unlock the global_row variable

      row_max = (row+CHUNK_SIZE) > N ? N : (row+CHUNK_SIZE); // calculate the maximum row of the chunk

      for (; row < row_max; row++){ // iterate through the chunk
        multiplier = A[row][norm] / A[norm][norm];
        for (int col = norm; col < N; col++){ // iterate through the columns
          A[row][col] -= A[norm][col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
      }
    }
  pthread_barrier_wait(&barrier); // wait for all threads to finish the current iteration
  }
}


void gauss() {
  int norm, row, col;  /* Normalization row, and zeroing element row and col */ 
  pthread_t thread[NTHREADS]; 
  
  printf("Computing Parallelly with Pthreads.\n");

  pthread_mutex_init(&global_row_lock,NULL); // initialize the global_row_lock
  pthread_barrier_init(&barrier, NULL, NTHREADS); // initialize the barrier
  
  for(int i=0; i < NTHREADS; ++i) {
    pthread_create(&thread[i], NULL, &gauss_thread, (void*)i); // create the threads
  }

  for(int i=0; i<NTHREADS; ++i){
    pthread_join( thread[i], NULL); // join the threads
  }

  pthread_mutex_destroy(&global_row_lock); // destroy the global_row_lock
  
  /* Back substitution */
  for (row = N - 1; row >= 0; row--){
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }

  printf("NTHREADS: %d\n", NTHREADS);

}