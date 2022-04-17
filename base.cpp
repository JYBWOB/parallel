#include<iostream>
#include<vector>
#include<cmath>
#include<stdio.h>
#include<cstring>
#include<windows.h>
#include<immintrin.h>
#include<fstream>
using namespace std;
const int maxN = 8;
//初始化A为一个N*N的矩阵的对称矩阵
float A[maxN][maxN] = {0};

//初始化数组b
float b[maxN] = {1.0};

 //初始化残差r,结果x,计算方向向量d
float r[maxN] = {-1};
float d[maxN] = {0};
float x[maxN] = {0};

void displayMatrix(float a[maxN][maxN], int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<a[i][j]<<"    ";
        }
        cout<<endl;
    }
}

void displayVector(float b[maxN], int N){
    for(int i=0;i< N;i++){
        cout<<b[i]<<" ";
    }
    cout<<endl;
}

//计算内积
float INNER_PRODUCT(float a[maxN], float b[maxN], int N){
    float res = 0;
    for(int i=0;i<N;i++){
        res+=a[i]*b[i];
    }
    return res;
}

//更新残差 r = A*x-b
void  MATRIX_VECTOR_PRODUCT(float *r, float a[maxN][maxN], float x[maxN],float b[maxN], int N){
    float temp = 0;
    for(int i=0;i<N;i++){
        temp = 0;
        for(int j=0;j<N;j++){
            temp += a[i][j]*x[j];
        }
        r[i] = temp - b[i];
    }
}

//计算dtAd
float MATRIX_PRODUCT(float a[maxN][maxN], float d[maxN], int N){
    float res = 0;
    float temp = 0;
    for(int i=0;i<N;i++){
        temp = 0;
        for(int j = 0;j<N;j++){
            temp += d[j]*A[i][j];
        }
        res += temp*d[i];
    }
    return res;
}


int main(){
    int N = 8;
    float total_time = 0.0f;
    //初始化A
    for(int i=0;i<N;i++){
        for(int j =0;j<N;j++){
            if(i==j){
                A[i][j] = 2;
            }
            if(abs(i-j) == 1){
                A[i][j] = -1;
            }
        }
    }
    MATRIX_VECTOR_PRODUCT(r, A, x, b, N);
    for(int i = 0; i < N; i++)
        d[i] = -r[i];
    
    displayMatrix(A, N);
    //displayVector(b, N);

    long long head , tail , freq ;
    QueryPerformanceFrequency ((LARGE_INTEGER *)&freq );
    QueryPerformanceCounter ((LARGE_INTEGER *)&head );
    //开始迭代
    for(int i =0;i<1024;i++){
        float r2 = INNER_PRODUCT(r, r, N);
        float dtAd = MATRIX_PRODUCT(A, d, N);

        //计算步长
        float alpha = r2/dtAd;

        //修正x
        for(int j=0;j<N;j++){
            x[j] = x[j] + alpha*d[j];
            r[j] = r[j] + alpha*INNER_PRODUCT(A[j], d, N);
        }
        float r2n = INNER_PRODUCT(r, r, N);
        if(r2n < 1e-6)
            break;
        int beta = r2n / r2;
        for(int j=0; j < N; j++) {
            d[j] = -r[j] + beta * d[j];
        }
    }
    QueryPerformanceCounter ((LARGE_INTEGER *)&tail );
    total_time += (tail - head) * 1000.0 / freq;
    cout << N << " : " << total_time<< "ms" << endl;
    displayVector(x, N);
    return 0;
}
