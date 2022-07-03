#include"Header.cuh"


void CUDA_Printf()
{
	printf("This is CUDA Printf111\n");
}

__global__ void matrix_elementwise_add(double* a, const double* b, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		a[index] = a[index] + b[index];
	}
}
void vectorAdd(double a[], double b[], double c[], int size)
{
	//c=a+b, 多用了变量c传回去
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

void vectorAdd1(double a[], double b[],int size)
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
	cudaMemcpy(a, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);

	//释放设备中变量所占内存  
	cudaFree(dev_a);
	cudaFree(dev_b);
	return;
}

__global__ void logisC(double* a, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		if (a[index] < -100)a[index] = 100.0;
		else a[index] *= -1.0;
		
		a[index] = 1.0 / (1.0 + exp(a[index]));
	}
}
void Logistic_compute(double a[],double target[], int size)
{
	double* dev_a = 0;

	// 在GPU中为变量dev_a、dev_b、dev_c分配内存空间.  
	cudaMalloc((void**)&dev_a, size * sizeof(double));

	// 从主机内存复制数据到GPU内存中.  
	cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);

	int threads_per_block = 1024;
	int num_blocks = (size + threads_per_block - 1) / threads_per_block;

	// 启动GPU内核函数  
	logisC << < num_blocks, threads_per_block >> > (dev_a, size);

	// 采用cudaDeviceSynchronize等待GPU内核函数执行完成并且返回遇到的任何错误信息  
	cudaDeviceSynchronize();

	// 从GPU内存中复制数据到主机内存中  
	cudaMemcpy(target, dev_a, size * sizeof(double), cudaMemcpyDeviceToHost);

	//释放设备中变量所占内存  
	cudaFree(dev_a);
	return;
}