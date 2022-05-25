#include<iostream>
#include<vector>
#include<cmath>
#include<stdio.h>
#include<cstring>
#include<time.h>
#include<fstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

#define BLOCK_SIZE 32
#define BLOCK_LENGTH 128

const int maxN = 2048;
//A为一个N*N的矩阵的对称矩阵
float *A;
//数组b
float *b;
//残差r,结果x,计算方向向量d
float *r;
float *d;
float *x;
float *dtAdMatrix;
float *dtAdVector;


// int MAX_ITER_TIME = 5000;
// bool FIX_ITER_TIME = true;
int MAX_ITER_TIME = 500000;
bool FIX_ITER_TIME = false;

struct timespec sts, ets;

void displayMatrix(float *a, int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<a[i*N+j]<<"    ";
        }
        cout<<endl;
    }
}

void displayVector(float *b, int N){
    for(int i=0;i< N;i++){
        cout<<b[i]<<" ";
    }
    cout<<endl;
}

//计算内积
float INNER_PRODUCT(float *a, float *b, int N){
    float res = 0.0;
    for(int i = 0; i < N; i++) {
        res += a[i]*b[i];
    }
    return res;
    // __shared__ float res = 0;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row == 0 && col < N) {
    //     res += a[col] * b[col];
    // }
    // return res;
}

//更新残差 r = A*x-b
void  MATRIX_VECTOR_PRODUCT(float *r, float *a, float *x,float *b, int N){
    float temp = 0;
    for(int i=0;i<N;i++){
        temp = 0;
        for(int j=0;j<N;j++){
            temp += a[i * N + j]*x[j];
        }
        r[i] = temp - b[i];
    }
}

//计算 d转置 * a矩阵 * d
__global__ void MATRIX_PRODUCT(float* res, float *a, float *d, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("cuda(%d, %d): %f\n", row, col, a[row * N + col]);
    if(col < N && row < N)
        res[row * N + col] = d[row] * a[row * N + col] * d[col];
}

// 求和
__global__ void SUM_MATRIX(float* res, float *a, int N){
    extern __shared__ float sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 一个线程负责把一个元素从全局内存载入到共享内存
    float x = 0;
    if (i < N)
    {
        x = a[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();// 等待所有线程把自己负责的元素载入到共享内存
    // 块内进行合并操作，每次合并变为一半
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)// 控制只有某些线程才进行操作。
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        // wait until all threads in the block have updated their partial sums
        __syncthreads();
    }
    // 每个块的线程0负责存放块内求和的结果
    if (threadIdx.x == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}

void MATRIX_PRODUCT_CPU(float* res, float *a, float *d, int N){
    *res = 0.0;
    for(int i=0;i<N;i++){
        for(int j = 0;j<N;j++){
            *res += d[i]*a[i*N+j]*d[j];
        }
    }
}


int main(){
    cudaMallocManaged((void **) &A, sizeof(float)*maxN*maxN);
    cudaMallocManaged((void **) &b, sizeof(float)*maxN);
    cudaMallocManaged((void **) &r, sizeof(float)*maxN);
    cudaMallocManaged((void **) &d, sizeof(float)*maxN);
    cudaMallocManaged((void **) &x, sizeof(float)*maxN);
    cudaMallocManaged((void **) &dtAdMatrix, sizeof(float)*maxN*maxN);
    cudaMallocManaged((void **) &dtAdVector, sizeof(float)*maxN);
    fstream file("res_base.csv", ofstream::out);
    for(int N = 128; N <= maxN; N += 128) {
    // for(int N = 4; N <= 1024; N += 4) {
        //初始化A
        for(int i=0;i<N;i++){
            for(int j =0;j<N;j++){
                if(i==j){
                  A[i * N + j] = 2;
                }
                else if(abs(i-j) == 1){
                  A[i * N + j] = -1;
                }
                else {
                  A[i * N + j] = 0;
                }
            }
        }
        for(int i = 0; i < N; i++) {
            b[i] = 1.0;
            x[i] = 0;
        }
        MATRIX_VECTOR_PRODUCT(r, A, x, b, N);
        for(int i = 0; i < N; i++) {
            d[i] = -r[i];
        }
        // displayMatrix(A, N);
        // displayVector(b, N);
        // displayVector(x, N);
        // displayVector(r, N);
        // displayVector(d, N);
        
        //displayMatrix(A, N);
        //displayVector(b, N);
        // cudaMemcpy(d_A, h_A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_r, h_r, sizeof(float)*N, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_d, h_d, sizeof(float)*N, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_x, h_x, sizeof(float)*N, cudaMemcpyHostToDevice);

        unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        int count = 0;
        timespec_get(&sts, TIME_UTC);
        //开始迭代
        for(int i =0;i<MAX_ITER_TIME;i++){
            count++;
            float r2 = INNER_PRODUCT(r, r, N);
            // displayMatrix(A, N);
            // displayVector(d, N);

            MATRIX_PRODUCT<<<dimGrid, dimBlock>>>(dtAdMatrix, A, d, N);
            cudaDeviceSynchronize();
            int gridSize = (N * N + BLOCK_LENGTH - 1) / BLOCK_LENGTH;
            SUM_MATRIX<<<gridSize, BLOCK_LENGTH>>>(dtAdVector, dtAdMatrix, N * N);
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) 
            {
                fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
            cudaDeviceSynchronize();
            float dtAd = 0.0;
            for(int p = 0; p < gridSize; p++) {
                dtAd += dtAdVector[p];
            }

            // cout << "cuda:" << dtAd << endl;
            // float ppp;
            // MATRIX_PRODUCT_CPU(&ppp, A, d, N);
            // cout << "cpu:" << ppp << endl;
            // cin >> ppp;

            // float ppp;
            // MATRIX_PRODUCT_CPU(&dtAd, A, d, N);
            // cout << "cpu:" << dtAd << endl;
            // cin >> ppp;


            //计算步长
            float alpha = r2/dtAd;
            //修正x
            for(int j=0;j<N;j++){
                x[j] = x[j] + alpha*d[j];
                r[j] = r[j] + alpha*INNER_PRODUCT(&A[j*N], d, N);
            }
            float r2n = INNER_PRODUCT(r, r, N);
            if(!FIX_ITER_TIME && r2n < 1e-4)
                break;
            int beta = r2n / r2;
            for(int j=0; j < N; j++) {
                d[j] = -r[j] + beta * d[j];
            }
        }
        timespec_get(&ets, TIME_UTC);
        time_t dsec = ets.tv_sec - sts.tv_sec;
        unsigned long long dnsec = ets.tv_nsec - sts.tv_nsec;
        cout << N << "：\t" << dsec << "." << dnsec << "\t" << count << endl;
        file << N << "," << dsec << "." << dnsec << "," << count << endl;
        // displayVector(x, N);
    }
    file.close();
    return 0;
}
