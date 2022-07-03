#include "cuda_runtime.h"  
#include "device_launch_parameters.h"    
#include "stdio.h"
#include<ctime>
#include <stdlib.h>
#include<iostream>
#include<time.h>
using namespace std;

__global__ void matrix_elementwise_add(double* a, const double* b,int n) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        a[index] = a[index] + b[index];
    }
}
void vectorAdd( double a[], double b[], double c[], int size)
{
	double* dev_a = 0;
	double* dev_b = 0;

	// 在GPU中为变量dev_a、dev_b、dev_c分配内存空间.  
	cudaMalloc((void**)&dev_a, size * sizeof(double));
	cudaMalloc((void**)&dev_b, size * sizeof(double));

	// 从主机内存复制数据到GPU内存中.  
	cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

	int threads_per_block = 1024;
	int num_blocks = (size + threads_per_block - 1) / threads_per_block;

	// 启动GPU内核函数  
	matrix_elementwise_add << < num_blocks, threads_per_block >> > (dev_a, dev_b, size);

	// 采用cudaDeviceSynchronize等待GPU内核函数执行完成并且返回遇到的任何错误信息  
	cudaDeviceSynchronize();

	// 从GPU内存中复制数据到主机内存中  
	cudaMemcpy(c, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);

	//释放设备中变量所占内存  
	cudaFree(dev_a);
	cudaFree(dev_b);
	return;
}

//__global__ static void sumOfSquares(int* num, int* result, clock_t* time)
//{
//    //声明一块共享内存
//    extern __shared__ int shared[];
//    //表示目前的 thread 是第几个 thread（由 0 开始计算）
//    const int tid = threadIdx.x;
//    //表示目前的 thread 属于第几个 block（由 0 开始计算）
//    const int bid = blockIdx.x;
//    shared[tid] = 0;
//    //thread需要同时通过tid和bid来确定，同时不要忘记保证内存连续性
//    for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM ) 
//    {
//        shared[tid] += num[i];
//    }
//
//    //同步 保证每个 thread 都已经把结果写到 shared[tid] 里面
//    __syncthreads();
//
//    //树状加法
//    int offset = 1, mask = 1;
//    while (offset < THREAD_NUM)
//    {
//        if ((tid & mask) == 0)shared[tid] += shared[tid + offset];
//
//        offset += offset;
//        mask = offset + mask;
//        __syncthreads();
//
//    }
//
//    //计算时间,记录结果，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
//    if (tid == 0)
//    {
//        result[bid] = shared[0];
//    }
//}

int main()
{
	double a[] = { 1, 2, 3, 4 };
	double b[] = { 2, 3, 4, 5 };
	double c[4];
	int size = 4;
	clock_t startTime = clock();
	vectorAdd(a, b, c,size);
	#include<ctime>
	clock_t endTime = clock();
	cout << "整个程序用时：" << double(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	printf("c's value= ");
	for (int i = 0; i < 4; ++i)
		printf("%lf ", c[i]);
    return 0;

}