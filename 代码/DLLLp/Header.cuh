#ifndef HEADER_CUH
#define HEADER_CUH
#include<math.h>
#include <cuda_runtime.h>  
#include <device_launch_parameters.h>    
#include <stdio.h>
#include<algorithm>
using namespace std;
const int threads_per_block = 512;
const int THREAD_NUM = 256;
//const int threads_per_block = 512;
extern "C" __declspec(dllexport)  void vectorAdd( double a[], double b[], double c[], int size);
extern "C" __declspec(dllexport) void CUDA_Printf();
extern "C" __declspec(dllexport) void vectorAdd1(double a[], double b[], int size);
extern "C" __declspec(dllexport) void Logistic_compute(double a[], double target[], int size);

extern "C" __declspec(dllexport) void Convolve_compute(const double* data, const double* kernel, double* pad, double* target, int* size8);


//loss_cuda
extern "C" __declspec(dllexport)void Logloss_compute (double a[], double target[], int size);
extern "C" __declspec(dllexport)void Logloss_getjacobi(double a[], int size);
extern "C" __declspec(dllexport)void Softmax_compute(double a[], double tem_sum[], double target[], const int size, const int block_num);
extern "C" __declspec(dllexport) double CrossEntropyWithSoftMax_compute(double a1[], double a2[], double sum_tmp[], const int size, const int block_num);

//basefunc
extern "C" __declspec(dllexport) double C_sum(double* data, int data_size);
extern "C" __declspec(dllexport) void C_Matmul(const double* a, const double* b, double* c, const int r3, const int c3, const int c1);

#endif
