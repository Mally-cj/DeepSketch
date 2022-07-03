#include"Header.cuh"

__global__ void loglossCompute(double* a, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		double tem = a[index];
		if (tem < -100)tem = 100.0;
		else tem *= -1.0;
		a[index] = log(1 + exp(tem));
	}
}
void Logloss_compute(double a[], double target[], int size)
{
	double* dev_a = 0;
	cudaMalloc((void**)&dev_a, size * sizeof(double));

	cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);

	int threads_per_block = 1024;
	int num_blocks = (size + threads_per_block - 1) / threads_per_block;

	loglossCompute << < num_blocks, threads_per_block >> > (dev_a, size);

	cudaDeviceSynchronize();

	cudaMemcpy(target, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);

	//释放设备中变量所占内存  
	cudaFree(dev_a);
	return;
}
__global__ void loglossGetjacobi(double* a, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n){
		a[index] = a[index] > 100.0 ? 100.0: a[index];
		a[index] = -1.0 / (1.0 + exp(a[index]));;
	}
}
void Logloss_getjacobi(double*a, int size)
{
	double* dev_a = 0;

	// 在GPU中为变量dev_a、dev_b、dev_c分配内存空间.  
	cudaMalloc((void**)&dev_a, size * sizeof(double));

	// 从主机内存复制数据到GPU内存中.  
	cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);

	int num_blocks = (size + threads_per_block - 1) / threads_per_block;

	// 启动GPU内核函数  
	loglossGetjacobi << < num_blocks, threads_per_block >> > (dev_a, size);

	// 采用cudaDeviceSynchronize等待GPU内核函数执行完成并且返回遇到的任何错误信息  
	cudaDeviceSynchronize();

	// 从GPU内存中复制数据到主机内存中  
	cudaMemcpy(a, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);

	//释放设备中变量所占内存  
	cudaFree(dev_a);
	return;
}

__global__ static void softmaxCompute(double* data, double* result_he, int data_size)
{
    /*
    使用树状加法和并行计算和
    */
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ double shared[];
    static double  Adjust_number = 10000.0;
    //用于减少精度损失，因为cuda实际只支持float，对一些小数会损失

    shared[tid] = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        if (data[index] > 100.0)data[index] = 100.0;
        data[index] = exp(data[index]) * Adjust_number;
        shared[tid] = data[index];
    }

    __syncthreads();

    int offset = 1, mask = 1;
    double  y = 0;  //v,y用于精度损失
    double r;
    while (offset < THREAD_NUM)
    {
        if ((tid & mask) == 0)
        {
            y -= shared[tid + offset];
            r = shared[tid] - y;
            y = r - shared[tid] + y;
            shared[tid] = r;
        }
        offset += offset;
        mask = offset + mask;
        __syncthreads();
    }

    if (tid == 0)result_he[bid] = shared[0];
}

__global__ void ss(double* data, int n, int v)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        data[index] /= v;
    }
}

void Softmax_compute(double a[], double sum_tmp[], double target[], const int size, const int block_num)
{
    double* dev_a;
    double* dev_sum;
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_sum, block_num * sizeof(double));

    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    softmaxCompute << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (dev_a, dev_sum, size);

    cudaDeviceSynchronize();
    cudaMemcpy(sum_tmp, dev_sum, block_num * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0;
    double y = 0;
    for (int i = 0; i < block_num; ++i)
    {
        double r = 0;
        y -= sum_tmp[i];
        r = sum - y;
        y = (r - sum) + y;
        sum = r;
    }
    ss << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (dev_a, size, sum);
    cudaMemcpy(target, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);

    return;
}


__global__ static void CrossEntropyWithSoftMaxCompute(double* data1, double* data2,  int data_size)
{
    /*
    使用树状加法和并行计算和  self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))
    结果直接存到data1中
    */
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ double shared[];
    shared[tid] = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        shared[tid] = data2[index] * log(data1[index]);
        //printf("gpu index %d:%lf\n", index, shared[tid]);
    }

    __syncthreads();

    int offset = 1, mask = 1;
    double  y = 0;  //v,y用于精度损失
    double r;
    while (offset < THREAD_NUM)
    {
        if ((tid & mask) == 0)
        {
            y -= shared[tid + offset];
            r = shared[tid] - y;
            y = r - shared[tid] + y;
            shared[tid] = r;
        }
        offset += offset;
        mask = offset + mask;
        __syncthreads();
    }

    if (tid == 0)data1[bid] = shared[0];
}


double CrossEntropyWithSoftMax_compute(double a1[], double a2[], double sum_tmp[], const int size, const int block_num)
{
    double *dev_a1;
    double *dev_a2;
    cudaMalloc((void**)&dev_a1, size * sizeof(double));
    cudaMalloc((void**)&dev_a2, size * sizeof(double));


    cudaMemcpy(dev_a1, a1, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a2, a2, size * sizeof(double), cudaMemcpyHostToDevice);
   
    CrossEntropyWithSoftMaxCompute << < block_num, THREAD_NUM, THREAD_NUM * sizeof(double) >> > (dev_a1, dev_a2, size);

    cudaDeviceSynchronize();
    cudaMemcpy(sum_tmp, dev_a1, block_num * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0;
    double y = 0;
    for (int i = 0; i < block_num; ++i)
    {
        double r = 0;
        y -= sum_tmp[i];
        r = sum - y;
        y = (r - sum) + y;
        sum = r;
    }
    sum *= -1;

    cudaFree(dev_a1);
    cudaFree(dev_a2);
    return sum;

}