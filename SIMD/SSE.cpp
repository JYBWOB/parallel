#include<iostream>
#include<vector>
#include<cmath>
#include<stdio.h>
#include<cstring>
#include<windows.h>
#include<immintrin.h>
#include<fstream>
using namespace std;
const int maxN = 2048;
//初始化A为一个N*N的矩阵的对称矩阵
float A[maxN][maxN] = {0};

//初始化数组b
float b[maxN] = {1.0};

 //初始化残差r,结果x,计算方向向量d
float r[maxN] = {-1};
float d[maxN] = {0};
float x[maxN] = {0};

// int MAX_ITER_TIME = 5000;
// bool FIX_ITER_TIME = true;
int MAX_ITER_TIME = 500000;
bool FIX_ITER_TIME = false;

struct timespec sts, ets;

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
    // float res = 0;
    // for(int i=0;i<N;i++){
    //     res+=a[i]*b[i];
    // }
    // return res;
    __m128 t1, t2, temp;
    __m128 res = _mm_setzero_ps();
    for(int i = 0; i < N; i += 4) {
        t1 = _mm_load_ps(a + i);
        t2 = _mm_load_ps(b + i);
        temp = _mm_mul_ps(t1, t2);
        res = _mm_add_ps(temp, res);
    }
    float result[4];
    _mm_store_ps(result, res);
    return result[0] + result[1] + result[2] + result[3];
}

//更新残差 r = A*x-b
void  MATRIX_VECTOR_PRODUCT(float *r, float a[maxN][maxN], float x[maxN],float b[maxN], int N){
    // float temp = 0;
    // for(int i=0;i<N;i++){
    //     temp = 0;
    //     for(int j=0;j<N;j++){
    //         temp += a[i][j]*x[j];
    //     }
    //     r[i] = temp - b[i];
    // }
    for(int i = 0; i < N; i++){
        r[i] = INNER_PRODUCT(a[i], x, N) - b[i];
    }
}

//计算dtAd
float MATRIX_PRODUCT(float a[maxN][maxN], float d[maxN], int N){
    float res = 0;
    // float temp = 0;
    for(int i=0;i<N;i++){
        // temp = 0;
        // for(int j = 0;j<N;j++){
        //     temp += d[j]*A[i][j];
        // }
        // res += temp*d[i];
        res += d[i] * INNER_PRODUCT(a[i], d, N);
    }
    return res;
}


int main(){
    fstream file("res_SSE.csv", ofstream::out);
    for(int N = 128; N <= maxN; N += 128) {
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
        for(int i = 0; i < N; i++) {
            b[i] = 1.0;
            x[i] = 0;
        }
        MATRIX_VECTOR_PRODUCT(r, A, x, b, N);
        for(int i = 0; i < N; i++) {
            d[i] = -r[i];
        }
        
        //displayMatrix(A, N);
        //displayVector(b, N);

        int count = 0;
        long long head , tail , freq ;
        QueryPerformanceFrequency ((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter ((LARGE_INTEGER *)&head );
        //开始迭代
        for(int i =0;i<MAX_ITER_TIME;i++){
            count++;
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
            if(!FIX_ITER_TIME && r2n < 1e-4)
                break;
            int beta = r2n / r2;
            for(int j=0; j < N; j++) {
                d[j] = -r[j] + beta * d[j];
            }
        }
        QueryPerformanceCounter ((LARGE_INTEGER *)&tail );
        float time = (tail - head) * 1000.0 / freq;
        cout << N << "：\t" << time << "\t" << count << endl;
        file << N << "," << time << "," << count << endl;
        // displayVector(x, N);
    }
    file.close();
    return 0;
}
