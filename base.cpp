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
float b[maxN] = {1};

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
    fstream file("init.csv", ios::out);
    for(int N = 32; N <= 2048; N += 32) {
        float total_time = 0.0f;
        for(int time = 0; time < 10; time++) {
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
            for(int i = 0; i < N; i++)
            {
                b[i] = 1.0;
                r[i] = -1.0;
                d[i] = x[i] = 0.0;
            }
           displayA(A, N);
        //    displayb(b, N);

            long long head , tail , freq ;
            QueryPerformanceFrequency ((LARGE_INTEGER *)&freq );
            QueryPerformanceCounter ((LARGE_INTEGER *)&head );
            //开始迭代
            int count = 0;
            for(int i =0;i<N;i++){
                count++;

                //计算r^Tr,
                float denom1 = INNER_PRODUCT(r,r, N);
                MATRIX_VECTOR_PRODUCT(r,A,x,b, N);

                float num1 = INNER_PRODUCT(r,r, N);
                if(num1 < 0.000001){
                    break;
                }
                float temp = num1/denom1;
                //计算方向向量d
                for(int j = 0;j<N;j++){
                    d[j] = -r[j]+temp*d[j];
                }
                float num2 = INNER_PRODUCT(d, r, N);
                float denom2 = MATRIX_PRODUCT(A, d, N);

                //计算步长
                float  length = -num2/denom2;

                //修正x
                for(int j=0;j<N;j++){
                    x[j] = x[j]+ length*d[j];
                }
            }
            QueryPerformanceCounter ((LARGE_INTEGER *)&tail );
            total_time += (tail - head) * 1000.0 / freq;
        }
        total_time /= 10;
        cout << N << " : " << total_time<< "ms" << endl;
        file << N << ',' << total_time << "\n";
    //    cout<<"迭代次数: "<<count<<endl;
//        displayb(x, N);
    }
    file.close();
    return 0;
}
