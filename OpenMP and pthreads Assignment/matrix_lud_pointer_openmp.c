#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
int t, n;

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

int main (int argc, char const* argv[]){
    if (argc < 3){
        printf("Error: too few arguments");
    }
    t = atoi(argv[1]);
    n = atoi(argv[2]);
    int chunk_size = (int)(n/t);
    double max;
    int k_prime;

    printf("Number of Threads: %d and Matrix A Dimension: %d\n", t, n);

    //A 
    double *A[n], *A_start[n];
    //L
    double *L[n];
    //U
    double *U[n];
    //Permutation (pi)
    int pi[n];

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
    // #pragma omp parallel for num_threads(t) schedule(static,chunk_size) private(drand_buff)
    // for(int j=0;j<n;j++){
    //     srand48_r(j, &drand_buff);
    //     // printf("%d\n",omp_get_thread_num());
    //     pi[j] = j;
    //     // L[j] = (double*)malloc(n*sizeof(double));
    //     // U[j] = (double*)malloc(n*sizeof(double));
    //     // A[j] = (double*)malloc(n*sizeof(double));
    //     // A_start[j] = (double*)malloc(n*sizeof(double));

    //     double* l = (double*)malloc(n*sizeof(double));
    //     double* u = (double*)malloc(n*sizeof(double));
    //     double* a = (double*)malloc(n*sizeof(double));
    //     double* a_start = (double*)malloc(n*sizeof(double));

    //     // int i is private (defined locally)
    //     for(int i=0;i<n;i++){
    //         if (i>j){
    //             // U[j][i] = 0.0;
    //             u[i] = 0.0;
    //         }
    //         else if (i==j){
    //             // L[j][i] = 1.0;
    //             l[i] = 1.0;
    //         }
    //         else if (i<j){
    //             // L[j][i] = 0.0;
    //             l[i] = 0.0;
    //         }
    //         // drand48_r(&drand_buff,&A[j][i]);
    //         drand48_r(&drand_buff,&a[i]);
    //         // printf("%f ",A[j][i]);
    //         // A_start[j][i] = A[j][i];
    //         a_start[i] = a[i];
    //     }
    //     L[j] = l;
    //     U[j] = u;
    //     A[j] = a;
    //     A_start[j] = a_start;
    //     // printf("\n");
    // }
    double end_init = omp_get_wtime();
    
    printf("Initialisation time (Matrices A, L, U): %f \n",(end_init-start_init));
    
    double start_omp = omp_get_wtime();
    double max_loop_time = 0;
    double lu_loop_time = 0;
    double a_loop_time = 0;
    for(int k=0;k<n;k++){
        max = 0;
        double start_max_loop = omp_get_wtime();
        // #pragma omp parallel for num_threads(t) schedule(static,(int)(n-k)/t) 
        #pragma omp parallel num_threads(t)
        {
            double thread_max = 0;
            int thread_kprime = 0;
            #pragma omp for reduction(max:max) schedule(static, chunk_size) nowait
            for(int i=k;i<n;i++){
                // printf("max: %f",max);
                // printf("A[i][k]: %f",A[i][k]);
                // printf("|A[i][k]|: %f",fabs(A[i][k]));
                if (max<fabs(A[k][i])){
                    thread_max = fabs(A[k][i]);
                    max = fabs(A[k][i]);
                    thread_kprime = i;
                }
            }
            #pragma omp critical
            {
                if(max == thread_max){
                    k_prime = thread_kprime;
                }
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
        #pragma omp parallel for num_threads(t) schedule(static,chunk_size)
        for(int i=k+1;i<n;i++){
            L[k][i] = A[k][i]/U[k][k];
            U[i][k] = A[i][k];
        }
        double lu_loop_end = omp_get_wtime();
        lu_loop_time += lu_loop_end - lu_loop_start;

        double a_loop_start = omp_get_wtime();
        #pragma omp parallel for num_threads(t) schedule(static,chunk_size)
        for(int j=k+1;j<n;j++){
            for(int i=k+1;i<n;i++){
                A[j][i] = A[j][i] - L[k][i]*U[j][k];
            }
        }
        double a_loop_end = omp_get_wtime();
        a_loop_time += a_loop_end - a_loop_start;
    }
    printf("Max Loop Time: %f \n",max_loop_time);
    printf("LU Update Loop Time: %f \n",lu_loop_time);
    printf("A Update Loop Time: %f\n",a_loop_time);

    double end_omp = omp_get_wtime();
    printf("OpenMP Implementation time: %f \n",(end_omp-start_omp));
    // double start_verify = omp_get_wtime();
    // verify(n,A_start,L,U,pi);
    // double end_verify = omp_get_wtime();
    // printf("Verify: %f \n",(end_verify-start_verify));
    return 0;
}
