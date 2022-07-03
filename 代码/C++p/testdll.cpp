#include"../DLLLp/Header.cuh"  //头文件路径
#pragma comment (lib, "E:/software/Anaconda3/envs/pytorch/C++project/new 2/Py/from_other/DLLLp.lib")   
using namespace std;
#define DATA_SIZE 3
#define BLOCK_NUM 32
#define THREAD_NUM 256
void GenerateNumbers(double* number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = double(rand() % 10)/1;
    }
}
int main()
{
    // 3*5 5*2;
    double a[15], b[10],c[6],cc[6];
    int r3, c3, c1;
    r3 = 3;
    c3 = 2;
    c1 = 5;
    GenerateNumbers(a, 15); GenerateNumbers(b, 10);
  

    C_Matmul(a, b, c, 3, 2, 5);
    for (int i = 0; i < 6; ++i)printf("%lf ", c[i]); puts("\n");


    memset(cc, 0, sizeof(cc));
    for (int i = 0; i < r3; ++i)
    {
        for (int k = 0; k < c1; ++k)
        {
            double v = a[i * c1 + k];
            int loc3 = i * c3;
            int loc2 = k * c3;

            for (int j = 0; j < c3; ++j)
            {
                cc[loc3 + j] += v * b[loc2 + j];
            }
        }
    }
    for (int i = 0; i < 6; ++i)printf("%lf\n", cc[i]);



  
    return 0;
}


