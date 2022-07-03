#include"Header.cuh"
#include<cublas_v2.h>
const int BLOCK_NUM = 32;
__global__ static void sumAll(double * num, double* result, int data_size)
{
    /*
    使用树状加法和并行计算和
    */
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    extern __shared__ double shared[];
    shared[tid] = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < data_size; i += gridDim.x * blockDim.x) {
        shared[tid] += num[i] ;
    }
    __syncthreads();

    //cuda中树状加法
    int offset = 1, mask = 1;
    while (offset < THREAD_NUM)
    {
        if ((tid & mask) == 0)
        {
            shared[tid] += shared[tid + offset];
        }
        offset += offset;
        mask = offset + mask;
        __syncthreads(); //同步所有线程
    }

    //计算时间,记录结果，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    //printf("uu threadIdx.x=%d\n", threadIdx.x);
    if (tid == 0)
    {
        result[bid] = shared[0];
        //printf("res[%d]=%lf\n", bid,result[bid]);
    }
}
double C_sum(double* data,  int data_size)
{
    const int MAX_BLOCK_NUM = 64;
    const int BLOCK_NUM = min(MAX_BLOCK_NUM, (data_size + THREAD_NUM - 1) / THREAD_NUM);

    double* gpudata, * result;
    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果，time用来存储运行时间 )
    cudaMalloc((void**)&gpudata, sizeof(double) * data_size);
    cudaMalloc((void**)&result, sizeof(double) * BLOCK_NUM);

    cudaMemcpy(gpudata, data, sizeof(double) * data_size, cudaMemcpyHostToDevice);
    
    sumAll<< < BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (gpudata, result,data_size);


    double sum[MAX_BLOCK_NUM];
    cudaMemcpy(&sum, result, sizeof(double) * BLOCK_NUM, cudaMemcpyDeviceToHost);

    cudaFree(gpudata);
    cudaFree(result);

    double final_sum = 0;
    for (int i = 0; i < BLOCK_NUM; i++)final_sum += sum[i];
  
    return final_sum;
}

__global__ static void matmul(double* a, double* b,double *c ,const int r3,const int c3,const int c1)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / c3;
    const int column = idx % c3;
  
   //计算矩阵乘法
    if (row < r3 && column < c3)
    {
        double t = 0;
        double y = 0;
        double r;
        for (int i = 0; i < c1; ++i)
        {
            y-= a[row * c1 + i] * b[i * c3 + column];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
        c[idx] = t;
    }
}
void C_Matmul(const double* a, const double* b, double* c, const int r3, const int c3, const int c1)
{
    double* cuda_a, * cuda_b, * cuda_c;
    //cudaMalloc 取得一块显卡内存 
    cudaMalloc((void**)&cuda_a, sizeof(double) * r3*c1);
    cudaMalloc((void**)&cuda_b, sizeof(double) * c1*c3);
    cudaMalloc((void**)&cuda_c, sizeof(double) * r3*c3);

    int size = r3 * c3;
    const int block_num = (size + THREAD_NUM - 1) / THREAD_NUM;

    cudaMemcpy(cuda_a, a, sizeof(double) * r3*c1, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(double) * c1*c3, cudaMemcpyHostToDevice);
    matmul << < block_num, THREAD_NUM, 0 >> > (cuda_a, cuda_b, cuda_c, r3,c3,c1);

    cudaMemcpy(c, cuda_c, sizeof(double) *(r3*c3), cudaMemcpyDeviceToHost);

    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

}

__global__ void convolveCompute1(double* data, double* pad,int* size8)
{
    int r = size8[0], kr = size8[2], hkr = size8[4], pr = size8[6];
    int c = size8[1], kc = size8[3], hkc = size8[5], pc = size8[7];

    const int idx = blockIdx.x * THREAD_NUM + threadIdx.x;
    const int row = idx / c;
    const int column = idx % c;
    if(row<r &&column<c) pad[(row + hkr) * pc + (column+ hkc)] = data[row * c + column];
}
__global__ void convolveCompute2(double* pad, double* kernel,double* target, int* size8)
{
    int r = size8[0], kr = size8[2], hkr = size8[4], pr = size8[6];
    int c = size8[1], kc = size8[3], hkc = size8[5], pc = size8[7];

    const int idx = blockIdx.x * THREAD_NUM + threadIdx.x;
    const int row = idx / c;
    const int col = idx % c;
    if (row < r && col < c)
    {
        double sum = 0;
        double y = 0;
        double r;
        for (int a = 0; a < kr; ++a)
            for (int b = 0; b < kc; ++b)
            {
                y -= pad[(row + a) * pc + (col + b)] * kernel[a * kc + b];
                r = sum - y;
                y = (r - sum) + y;
                sum = r;
            }
        target[idx] = sum;
    }
}


void Convolve_compute(const double* data, const double* kernel,double* pad, double* target, int *size8)
{
    int r=size8[0], kr=size8[2], hkr=size8[4], pr=size8[6];
    int c=size8[1],kc=size8[3] , hkc=size8[5], pc=size8[7] ;
    
    double* dev_data, * dev_kernel, * dev_pad;
    int *dev_size8;
    cudaMalloc((void**)&dev_data, (r*c) * sizeof(double));
    cudaMalloc((void**)&dev_kernel, (kr*kc) * sizeof(double));
    cudaMalloc((void**)&dev_pad, (pr * pc) * sizeof(double));
    cudaMalloc((void**)&dev_size8, 8 * sizeof(int));

    cudaMemcpy(dev_data, data, (r*c)* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel, kernel, (kr*kc) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_size8, size8, 8 * sizeof(int), cudaMemcpyHostToDevice);
    
    int block_num = (r * c + THREAD_NUM - 1) / THREAD_NUM;
    convolveCompute1 << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (dev_data, dev_pad, dev_size8);

    cudaDeviceSynchronize();
    convolveCompute2 << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (dev_pad, dev_kernel, dev_data, dev_size8);

    cudaMemcpy(pad, dev_pad, (pr * pc) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(target, dev_data, (r * c) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_data); cudaFree(dev_size8);
    cudaFree(dev_kernel);
}



//void Softmax_compute(const double* data,const int data_size, const int block_num, double* sum_tmp, double* out )
//{
//    double* gpudata, * result_he;
//    cudaMalloc((void**)&gpudata, sizeof(double) * data_size);
//    cudaMalloc((void**)&result_he, sizeof(double) * block_num);
//   
//    cudaMemcpy(gpudata, data, sizeof(double) * data_size, cudaMemcpyHostToDevice);
//
//    softmaxCompute << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (gpudata, result_he, data_size);
//
//    //double sum2[100];
//    cudaMemcpy(&sum_tmp, result_he, sizeof(double) * block_num, cudaMemcpyDeviceToHost);
//    cudaMemcpy(&out, gpudata, sizeof(double) * data_size, cudaMemcpyDeviceToHost);
//    cudaFree(gpudata);
//    cudaFree(result_he);
//
//   
//  /*  for (int i = 0; i < block_num; ++i)
//    {
//        printf("i= %lf\n", sum2[i]);
//        sum += sum_tmp[i];
//    }*/
//   //ss << < BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (gpudata,  data_size,sum);
//
//   /* cudaMemcpy(&out, gpudata, sizeof(double) * data_size, cudaMemcpyDeviceToHost);
//    cudaFree(gpudata);
//    cudaFree(result_he);*/
//
//   
//
//    //double ans2 = C_sum(sum2, 1);
//
//    //for (int i = 0; i < BLOCK_NUM; i++)printf("sum=%lf  ans2=%lf\n", sum2[i],ans2);
//    //////计算和
//   
//    //sumAll << < 1, BLOCK_NUM, BLOCK_NUM * sizeof(double) >> > (result_he, he_all,1);
//    //double sum[1];
//    //cudaMemcpy(&sum, he_all, sizeof(double) * 1, cudaMemcpyDeviceToHost); 
//    //cudaDeviceSynchronize();
//
//    //printf("2:sum=%lf\n", sum[0]);
//    //printf("blocknum=%d\n", BLOCK_NUM);
//    //double sum;
//    ////cudaMemcpy(&out2, gpudata, sizeof(double) * data_size, cudaMemcpyDeviceToHost);
//    //cudaMemcpy(&sum, he_all, sizeof(double) * BLOCK_NUM, cudaMemcpyDeviceToHost);
//
//   
//
//    //printf("sum= %lf\n", sum);
//    //double final_sum = 0;
//    //for (int i = 0; i < BLOCK_NUM; i++)final_sum += sum[i];
//
//    //cudaDeviceSynchronize();
//    //printf("sizeof  %d\n", sizeof(out));
//    //for (int i = 0; i < data_size; ++i)
//    //{
//    //    //printf("data[%d]=%lf\n", i, data[i]);
//    //    printf("out[%d]=%lf\n", i, out[i]);
//    //    printf("out2[%d]=%lf\n", i, out2[i]);
//    //    //out[i] = out2[i] / final_sum;
//
//    //}
//}



//void Softmax_compute(double a[], double tem_sum[], double target[], const int size, const int block_num)
//{
//     double* dev_a;
//        double* dev_sum;
//        cudaMalloc((void**)&dev_a, size * sizeof(double));
//        cudaMalloc((void**)&dev_sum, block_num * sizeof(double));
//    
//    
//        cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
//        softmaxCompute << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (dev_a, dev_sum, size);
//        
//        cudaDeviceSynchronize();
//        cudaMemcpy(tem_sum, dev_sum, block_num * sizeof(double), cudaMemcpyDeviceToHost);
//        cudaMemcpy(target, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);
//
//        cudaFree(dev_a);
//        double sum = 0;
//        for (int i = 0; i < block_num; ++i)sum += tem_sum[i];
//        for (int i = 0; i < size; ++i)target[i] /= sum;
//        printf("sum=%lf\n", sum);
//        return;
//}


