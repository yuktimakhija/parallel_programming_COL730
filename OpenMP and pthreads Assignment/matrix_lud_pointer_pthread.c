#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
int t, n;
//A 
double **A;
//L
double **L;
//U
double **U;

void swap(double *a, double*b){
    double temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

void matrix_multiply(int n, double *A[n], double *B[n], double *C[n]){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            C[j][i] = 0;
            for(int k=0;k<n;k++){
                C[j][i] += A[k][i]*B[j][k];
            }
        }
    }
}

void verify(int n, double *A[n], double *L[n], double *U[n], int *pi){
    // printf("Pi:\n");
    double *P[n];
    double *PA[n], *LU[n], *residual[n];
    for(int p=0;p<n;p++){
        // printf("%d ",pi[p]);
        PA[p] = (double*)malloc(n*sizeof(double));
        LU[p] = (double*)malloc(n*sizeof(double));
        P[p] = (double*)malloc(n*sizeof(double));
        residual[p] = (double*)malloc(n*sizeof(double));
    }
    for(int p=0;p<n;p++){
        for(int p2=0;p2<n;p2++){
            P[p2][p] = 0;
        }
        P[pi[p]][p] = 1;
    }
    double start_mult = omp_get_wtime();
    matrix_multiply(n,P,A,PA);
    matrix_multiply(n,L,U,LU);
    double end_mult = omp_get_wtime();
    printf("matrix multiplication (in verify): %f \n",(end_mult-start_mult));
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            residual[j][i] = PA[j][i] - LU[j][i];
        }
    }
    double l21_norm = 0.0; 
    for (int i=0;i<n;i++){    
        double l2_norm = 0.0;
        for (int j=0;j<n;j++){
            l2_norm += pow(residual[i][j], 2);
        }
        l21_norm += sqrt(l2_norm);
    }
    printf("L21 norm %f\n",l21_norm);
}

struct matrices_args {
    int n, k, s, e;
    // double *A[n], *L[n], *U[n];
};

void* update_A(void *input_p){
    struct matrices_args *input = input_p;
    // printf("From update_A: s=%d, e=%d\n",input->s,input->e);
    for(int j=input->s;j<input->e;j++){
        for(int i=input->k+1;i<input->n;i++){
            A[j][i] = A[j][i] - L[input->k][i]*U[j][input->k];
        }
    }
}

void* update_LU(void *input_p){
    struct matrices_args *input = input_p;
    // printf("From update_LU: s=%d, e=%d\n",input->s,input->e);
    for(int i=input->s;i<input->e;i++){
        L[input->k][i] = A[input->k][i]/U[input->k][input->k];
        U[i][input->k] = A[i][input->k];
    }
}


int main (int argc, char const* argv[]){
    if (argc < 3){
        printf("Error: too few arguments");
    }
    t = atoi(argv[1]);
    n = atoi(argv[2]);
    pthread_t thread_handles[t];
    int chunk_size = (int)(n/t);
    double max;
    int k_prime;

    printf("Number of Threads: %d and Matrix A Dimension: %d\n", t, n);

    double *A_start[n];
    //Permutation (pi)
    int pi[n];

    A = (double**)malloc(n*sizeof(double*));
    U = (double**)malloc(n*sizeof(double*));
    L = (double**)malloc(n*sizeof(double*));
    // OpenMP Implementation 
    // srand48(time(0));

    // struct drand48_data drand_buff;
    double start_init = omp_get_wtime();
    srand48(time(0));
    for(int j=0;j<n;j++){
        pi[j] = j;

        L[j] = (double*)malloc(n*sizeof(double));
        U[j] = (double*)malloc(n*sizeof(double));
        A[j] = (double*)malloc(n*sizeof(double));
        A_start[j] = (double*)malloc(n*sizeof(double));

        for(int i=0;i<n;i++){
            if (i>j){
                U[j][i] = 0.0;
            }
            else if (i==j){
                L[j][i] = 1.0;
            }
            else if (i<j){
                L[j][i] = 0.0;
            }
            A[j][i] = drand48();
            A_start[j][i] = A[j][i];
        }
    }
    
    double end_init = omp_get_wtime();

    printf("Initialisation time (Matrices A, L, U): %f \n",(end_init-start_init));
    
    double start_omp = omp_get_wtime();
    double max_loop_time = 0;
    double lu_loop_time = 0;
    double a_loop_time = 0;
    for(int k=0;k<n;k++){
        double start_max_loop = omp_get_wtime();
        max = 0;
        for(int i=k;i<n;i++){
            // printf("max: %f",max);
            // printf("A[i][k]: %f",A[i][k]);
            // printf("|A[i][k]|: %f",fabs(A[i][k]));
            if (max<fabs(A[k][i])){
                max = fabs(A[k][i]);
                k_prime = i;
            }
        }
        
        double end_max_loop = omp_get_wtime();
        max_loop_time += (end_max_loop-start_max_loop);
        
        // printf("max: %f",max);
        // printf("kprime: %d",k_prime);
        if (max == 0){
            printf("Error: singular matrix");
            return 0;
        }
        // printf("Swapping the rows\n");
        // Swap k and k_prime in pi
        int temp = pi[k];
        pi[k] = pi[k_prime];
        pi[k_prime] = temp;
        // Swap row k with k_prime
        for(int a=0;a<n;a++){
            swap(&A[a][k],&A[a][k_prime]);
            if(a<k){
                swap(&L[a][k],&L[a][k_prime]);
            }
        }
        U[k][k] = A[k][k];
    

        double lu_loop_start = omp_get_wtime();
        // k+1 to n, k+1+thread*work
        // work = (n-(k+1) +1)/t
        int work = (int) (n-(k+1))/t;
        for (int thread = 0; thread < t; thread++){
            struct matrices_args *args = (struct matrices_args*) malloc(sizeof(struct matrices_args));
            // memcpy(args->A, A, n*n*sizeof(double));
            // memcpy(args->n,n,sizeof(int));
            // memcpy(args->k,k,sizeof(int));
            // args->A = A;
            args->n = n;
            args->k = k;
            args->s = k+ 1 + thread*work;
            args->e = args->s + work;
            if ((thread == t-1) && (args->e != n)) {
                args->e = n;
            }
            // printf("From main (LU loop): s=%d, e=%d\n",args->s,args->e);
            // args->L = L;
            // args->U = U;
            pthread_create(&thread_handles[thread], NULL, update_LU, (void *)args);
        }
        for (int thread = 0; thread < t; thread++){
            pthread_join(thread_handles[thread], NULL);
        }
        double lu_loop_end = omp_get_wtime();
        lu_loop_time += lu_loop_end - lu_loop_start;

        double a_loop_start = omp_get_wtime();
        for (int thread = 0; thread < t; thread++){
            struct matrices_args *args = (struct matrices_args*) malloc(sizeof(struct matrices_args));
            // memcpy(args->A, A, n*n*sizeof(double));
            // memcpy(args->n,n,sizeof(int));
            // memcpy(args->k,k,sizeof(int));
            // args->A = A;
            args->n = n;
            args->k = k;
            args->s = k+ 1 + thread*work;
            args->e = args->s + work;
            if ((thread == t-1) && (args->e != n)) {
                args->e = n;
            }
            // args->L = L;
            // args->U = U;
            // printf("From main (A loop): s=%d, e=%d\n",args->s,args->e);
            pthread_create(&thread_handles[thread], NULL, update_A, (void *)args);
        }
        for (int thread = 0; thread < t; thread++){
            pthread_join(thread_handles[thread], NULL);
        }
        double a_loop_end = omp_get_wtime();
        a_loop_time += a_loop_end - a_loop_start;
    }
    printf("Max Loop Time: %f \n",max_loop_time);
    printf("LU Update Loop Time: %f \n",lu_loop_time);
    printf("A Update Loop Time: %f\n",a_loop_time);

    double end_omp = omp_get_wtime();
    printf("p-threads Implementation time: %f \n",(end_omp-start_omp));
    // double start_verify = omp_get_wtime();
    // verify(n,A_start,L,U,pi);
    // double end_verify = omp_get_wtime();
    // printf("Verify: %f \n",(end_verify-start_verify));
    return 0;
}