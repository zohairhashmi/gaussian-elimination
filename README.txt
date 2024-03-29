README - HOMEWORK 2
Name: Zohair Hashmi
UIN: 668913771
Date: 02/25/2024

System:
- Chameleon.org
- Project: CHI-241240 CHI@UC
- Image name: CC-CentOS8-stream-CUDA11
- IP Address: 192.5.87.89


Steps to execute each code file:

1. Edit code & make changes to NTHREADS for testing stage on local machine.

2. Upload the codes to the Virtual Machine using the following command:
sudo scp -i </path/.pem file for permissions> -r </path/.c file to upload>  cc@192.5.87.89:test-folder

3. SSH into the VM using the following linux command:
sudo ssh -i /mnt/e/cs566_zh.pem cc@192.5.87.89

4. compile the code using following commands for each Sequential, Pthread, OpenMP & MPI tests:
gcc gauss.c -o gauss.out
gcc -pthread gauss-pthread.c -o pthreadgauss.out
gcc -fopenmp gauss-openmp.c -o ompgauss.out
mpicc -o gauss-mpi gauss-mpi.c 

5. run the executable file using following command:
./a.out <N> <SEED>
mpiexec -n 1 ./gauss-mpi <N> <SEED>
