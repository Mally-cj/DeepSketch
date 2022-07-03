#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

//1M
#define DATA_SIZE 10489

#define THREAD_NUM 256

#define BLOCK_NUM 32

int data[DATA_SIZE];

//产生大量0-9之间的随机数
void GenerateNumbers(int* number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}

//打印设备信息
void printDeviceProp(const cudaDeviceProp& prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;

    for (i = 0; i < count; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //打印设备信息
        printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}


// __global__ 函数 (GPU上执行) 计算立方和
__global__ static void sumOfSquares(int* num, int* result)
{
    //声明一块共享内存
    extern __shared__ int shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    shared[tid] = 0;
    //thread需要同时通过tid和bid来确定，同时不要忘记保证内存连续性
    for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        shared[tid] += num[i] * num[i] * num[i];
    }

    //同步 保证每个 thread 都已经把结果写到 shared[tid] 里面
    __syncthreads();

    //使用线程0完成加和
    if (tid == 0)
    {
        for (int i = 1; i < THREAD_NUM; i++)
        {
            shared[0] += shared[i];
        }
        result[bid] = shared[0];
    }
}

// __global__ 函数 (GPU上执行) 计算立方和
__global__ static void sumOfSquares1(int* num, int* result)
{
    /*
    使用树状加法和并行计算和
    */
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    shared[tid] = 0;

    for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
        shared[tid] += num[i] * num[i] * num[i];
    }
    __syncthreads();

    //树状加法
    int offset = 1, mask = 1;
    while (offset < THREAD_NUM)
    {
        if ((tid & mask) == 0)
        {
            shared[tid] += shared[tid + offset];
        }
        offset += offset;
        mask = offset + mask;
        __syncthreads();
    }

    //计算时间,记录结果，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    if (tid == 0)
    {
        result[bid] = shared[0];
    }
}



int main()
{
    //生成随机数
    GenerateNumbers(data, DATA_SIZE);

    /*把数据复制到显卡内存中*/
    int* gpudata, * result;

    clock_t* time;

    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果，time用来存储运行时间 )
    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int) * BLOCK_NUM);

    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfSquares1 << < BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int) >> > (gpudata, result);

    int *sum = (int*)malloc(sizeof(int) * BLOCK_NUM);

    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);

    //Free
    cudaFree(gpudata);
    cudaFree(result);

    int final_sum = 0;
    printf("len %d\n", sizeof(sum));
    for (int i = 0; i < BLOCK_NUM; i++) {

        //final_sum += sum[i];

    }
    free(sum);
    //sum = NULL;

    printf("GPUsum: %d \n", final_sum);

    final_sum = 0;

    for (int i = 0; i < DATA_SIZE; i++) {

        final_sum += data[i] * data[i] * data[i];

    }

    printf("CPUsum: %d \n", final_sum);

    return 0;
}