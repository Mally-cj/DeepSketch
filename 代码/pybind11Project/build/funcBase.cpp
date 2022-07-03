#include"func.h"

py::array_t<double> matAdd(py::array_t<double>& arr_a, py::array_t<double>& arr_b) {

	py::buffer_info bufA = arr_a.request(), bufB = arr_b.request();
	//request方法活得对py::array_t<T>的绑定，包括维度、数据指针、size、shape等参数

	auto result = py::array_t<double>({ bufA.shape[0],bufA.shape[1] });

	py::buffer_info bufResult = result.request();
	double* ptrA = (double*)bufA.ptr,
		* ptrB = (double*)bufB.ptr,
		* ptrResult = (double*)bufResult.ptr;  //获得数据指针

	vectorAdd(ptrA, ptrB, ptrResult, bufA.shape[0] * bufA.shape[1]);

	/*for (int i = 0; i < size; ++i)
		py::print("he:", i, "  ", ptrA, ptrB, ptrResult[i]);*/

	return result;
}
py::array_t<double> matAdd1(py::array_t<double>& arr_a, py::array_t<double>& arr_b) {

	py::buffer_info bufA = arr_a.request(), bufB = arr_b.request();
	//request方法活得对py::array_t<T>的绑定，包括维度、数据指针、size、shape等参数

	double* ptrA = (double*)bufA.ptr;
	double* ptrB = (double*)bufB.ptr;  //获得数据指针

	vectorAdd1(ptrA, ptrB, bufA.shape[0] * bufA.shape[1]);
	return arr_a;
}


py::array_t<double> change2(py::array_t<double> &input)
{
    /*
     把传入的矩阵都变成行主序
    */

    py::buffer_info buf = input.request();
    double* ptr = (double*)buf.ptr;
    const int unit_r = buf.strides[0] / sizeof(ptr);
    const int unit_c = buf.strides[1] / sizeof(ptr);

    //保证是行主序
 /*   if (unit_r >= unit_c && unit_c == 1)
        return  input;*/

    auto result = py::array_t<double>({ buf.shape[0] ,buf.shape[1] });
    py::buffer_info buf1 = result.request();
    double* ptr1 = (double*)buf1.ptr;

    for (int i = 0, t = 0; i < buf.size; ++i, t += unit_r)ptr1[i] = ptr[t];

    return result;
}
py::array_t<double> matmul_2d(py::array_t<double>& input1, py::array_t<double>& input2)
{
    /*
    2维矩阵相乘，朴素算法
    */
    input1 = change2(input1);
    input2 = change2(input2);

    // 获取input1, input2的信息
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    if (buf1.ndim != 2 || buf2.ndim != 2)
    {
        throw std::runtime_error("Number of dimensions must be two！");
    }
    if (buf1.shape[1] != buf2.shape[0])
    {
        throw std::runtime_error("Input shape must match");
    }
    const int buf1_r = buf1.shape[0];
    const int buf1_c = buf1.shape[1];
    const int buf2_r = buf2.shape[0];
    const int buf2_c = buf2.shape[1];
    const int buf3_r = buf1.shape[0];
    const int buf3_c = buf2.shape[1];

    //申请空间,确定结果矩阵的形状
    auto result = py::array_t<double>({ buf3_r,buf3_c });
    py::buffer_info buf3 = result.request();

    //获取numpy.ndarray 数据指针
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr3 = (double*)buf3.ptr;

    int loc3 = 0;
    for (int i = 0; i < buf3_r; ++i)
    {
        for (int j = 0; j < buf3_c; ++j, ++loc3)
        {
            double sum = 0;
            int loc2 = j;
            int loc1 = i * buf1_c;
            for (int k = 0; k < buf1_c; ++k, ++loc1, loc2 += buf2_c)
            {
                //是矩阵 ptr1(i,k)*ptr2（k,j）
                sum += ptr1[loc1] * ptr2[loc2];
                //py::print("sum", i,j,k, ptr1[loc1], ptr2[k * unit2_r + j * unit2_c]);
            }
            ptr3[loc3] = sum;
        }
    }
    return result;
}
py::array_t<double> matmul_2d2(py::array_t<double>& input1, py::array_t<double>& input2)
{
    /*
     2d矩阵相乘，调换循环顺序的算法
    */
    input1 = change2(input1);
    input2 = change2(input2);

    // 获取input1, input2的信息
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
    {
        throw std::runtime_error("Number of dimensions must be two！");
    }

    if (buf1.shape[1] != buf2.shape[0])
    {
        throw std::runtime_error("Input shape must match");
    }
    const  int buf1_r = buf1.shape[0];
    const  int buf1_c = buf1.shape[1];
    const int buf2_r = buf2.shape[0];
    const int buf2_c = buf2.shape[1];
    const int buf3_r = buf1.shape[0];
    const int buf3_c = buf2.shape[1];

    //申请空间,确定结果矩阵的形状
    auto result = py::array_t<double>({ buf3_r,buf3_c });
    py::buffer_info buf3 = result.request();

    //获取numpy.ndarray 数据指针
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr3 = (double*)buf3.ptr;
    memset(ptr3, 0, sizeof(ptr3) * (buf3_c * buf3_r));



    int loc1 = 0;
    //指针访问numpy.ndarray
    for (int i = 0; i < buf3_r; ++i)
    {
        int loc2 = 0;
        for (int k = 0; k < buf1_c; ++k, ++loc1)
        {
            double v = ptr1[loc1];
            int loc3 = buf3_r * i;
            for (int j = 0; j < buf3_c; ++j, ++loc2, ++loc3)
            {
                ptr3[loc3] += v * ptr2[loc2];
            }
            
        }
    }
    for(int i=0;i<buf3_r;++i)
        for(int j=0;j<buf3_c;++j)
            py::print("ptr3", i, j, ptr3[buf3_c*i+j]);

    return result;
}

py::array_t<double> matmul(const py::array_t<double>& aa, const py::array_t<double>& bb,bool gpu)
{
    // 获取input1, input2的信息
    py::buffer_info buf1 = aa.request();
    py::buffer_info buf2 = bb.request();


    const int buf1_r = buf1.shape[0];
    const int buf1_c = buf1.shape[1];
    const int buf2_r = buf2.shape[0];
    const int buf2_c = buf2.shape[1];
    const int buf3_r = buf1.shape[0];
    const int buf3_c = buf2.shape[1];

    auto result = py::array_t<double>({ buf3_r,buf3_c });
    py::buffer_info buf3 = result.request();

    //获取numpy.ndarray 数据指针
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr3 = (double*)buf3.ptr;
    memset(ptr3, 0, sizeof(ptr3) * (buf3_c * buf3_r));

    if (gpu == false)
    {
        for (int i = 0; i < buf3_r; ++i)
        {
            for (int k = 0; k < buf1_c; ++k)
            {
                double v = ptr1[i * buf1_c + k];
                int loc3 = i * buf3_c;
                int loc2 = k * buf2_c;
                for (int j = 0; j < buf3_c; ++j)
                {
                    ptr3[loc3 + j] += v * ptr2[loc2 + j];
                }
            }
        }
    }
    else
    {
        C_Matmul(ptr1, ptr2, ptr3, buf3_r, buf3_c, buf1_c);
    }
    return result;
}