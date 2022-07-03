#include"func.h"

py::array_t<double>  C_Logistic::compute( py::array_t<double>& x, int gpu )
{
    ////self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))
    
    //part1
    py::buffer_info bufx = x.request();
    double* ptrx = (double*)bufx.ptr;
   

    //part2  把value_ptr绑到这次计算的结果上，即python里的value=compute()
    auto result = py::array_t<double>({ bufx.shape[0],bufx.shape[1] });
    py::buffer_info bufResult = result.request();
    value_ptr= (double*)bufResult.ptr;
    value_len = bufx.shape[0]*bufx.shape[1];

    //part2
    if (gpu == 0) {
        int size = bufx.shape[0] * bufx.shape[1];
        for (int i = 0; i < size; ++i)
        {
            if (ptrx[i] < -100)ptrx[i] = 100.0;
            else
                ptrx[i] *= -1;

            //double tem = exp(ptrx[i]);
            value_ptr[i] = 1.0 / (1.0 + exp(ptrx[i]));
        }
    }
    else {
        Logistic_compute(ptrx, value_ptr, bufx.shape[0]*bufx.shape[1]);
    }
    return result;
}

py::array_t<double>  C_Logistic::getjacobi()
{
    /*
    python code: return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)   
    */
    auto result = py::array_t<double>({value_len, value_len});
    py::buffer_info bufResult = result.request();
    double* ptr = (double*)bufResult.ptr;

    int index = 0;
    for (int i = 1; i <= value_len; ++i)
    {
        ptr[index] = value_ptr[i] * (1 - value_ptr[i]);
        index += i;
        index += value_len;
    }
    return result;
}

py::array_t<double>  C_Convolve::compute(py::array_t<double>& data, py::array_t<double>& kernel, bool gpu)
{

    py::buffer_info buf1 = data.request();
    double* ptr1 = (double*)buf1.ptr;

    py::buffer_info buf2 = kernel.request();
    double* ptr2 = (double*)buf2.ptr;

    r = buf1.shape[0];
    c = buf1.shape[1];
    kr = buf2.shape[0];
    kc = buf2.shape[1];

    hkr = int(kr / 2);
    hkc = int(kc / 2);

    pr = r + hkr * 2;
    pc = c + hkc * 2;
   
    py::array_t<double> padded = py::array_t<double>({ pr,pc});
    py::buffer_info buf3 = padded.request();
    padded_ptr= (double*)buf3.ptr;

    py::array_t<double> out = py::array_t<double>({ r,c });
    py::buffer_info buf4 = out.request();
    double* ptr = (double*)buf4.ptr;

    memset(padded_ptr, 0, sizeof(padded_ptr)*(pr*pc));

    if (gpu == false)
    {
        //实现这个self.padded[hkw:hkw+w, hkh:hkh+h]=data
        for (int i = 0; i < r; ++i)
            for (int d = 0; d < c; ++d)
            {
                //(i+hkr,d+hkc) ---> （i+hkr）*pc+(d+hkc)
                //（i,d)---> i*c+d
                padded_ptr[(i + hkr) * pc + (d + hkc)] = ptr1[i * c + d];
            }

        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
            {
                double sum = 0;
                double y = 0;
                double r;
                for (int a = 0; a < kr; ++a)
                    for (int b = 0; b < kc; ++b)
                        //写成二维就是sum+=padded_ptr[i+a][j+b]*kernel[a][b]
                    {
                        y-= padded_ptr[(i + a) * pc + (j + b)] * ptr2[a * kc + b];
                     
                        r = sum - y;
                        y = (r - sum) + y;
                        sum = r;
                    }
                ptr[i * c + j] = sum;
            }
    }
    else if(gpu==true)
    {
        int size[8];
        size8[0] = r; size8[2] = kr; size8[4] = hkr; size8[6] = pr;
        size8[1] = c; size8[3] = kc; size8[5] = hkc; size8[7] = pc;
        Convolve_compute(ptr1,ptr2, padded_ptr,ptr,size8);
    }
    return out;
}

py::array_t<double>  C_Convolve::compute2(py::array_t<double>& data, py::array_t<double>& kernel, bool gpu)
{
    //py::array_t<double> kernel2 = change2(kernel);


    py::buffer_info buf1 = data.request();
    double* ptr1 = (double*)buf1.ptr;

    py::buffer_info buf2 = kernel.request();
    double* ptr2 = (double*)buf2.ptr;

    int unit_r = buf2.strides[0] / sizeof(ptr2);
    int unit_c = buf2.strides[1] / sizeof(ptr2);
    py::print("helo", unit_r, unit_c);
    py::print("stide", buf2.strides[0], buf2.strides[1]);

    r = buf1.shape[0];
    c = buf1.shape[1];
    kr = buf2.shape[0];
    kc = buf2.shape[1];

    hkr = int(kr / 2);
    hkc = int(kc / 2);

    pr = r + hkr * 2;
    pc = c + hkc * 2;

    py::array_t<double> padded = py::array_t<double>({ pr,pc });
    py::buffer_info buf3 = padded.request();
    padded_ptr = (double*)buf3.ptr;

    py::array_t<double> out = py::array_t<double>({ r,c });
    py::buffer_info buf4 = out.request();
    double* ptr = (double*)buf4.ptr;

    memset(padded_ptr, 0, sizeof(padded_ptr)*pr*pc);

    if (gpu == false)
    {
        //实现这个self.padded[hkw:hkw+w, hkh:hkh+h]=data
        for (int i = 0; i < r; ++i)
            for (int d = 0; d < c; ++d)
            {
                //(i+hkr,d+hkc) ---> （i+hkr）*pc+(d+hkc)
                //（i,d)---> i*c+d
                padded_ptr[(i + hkr) * pc + (d + hkc)] = ptr1[i * c + d];
            }

        py::print("prepare");
        return padded;
        py::print("here");

        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
            {
                double sum = 0;
                double y = 0;
                double r;
                for (int a = 0; a < kr; ++a)
                    for (int b = 0; b < kc; ++b)
                        //写成二维就是sum+=padded_ptr[i+a][j+b]*kernel[a][b]
                    {
                        y -= padded_ptr[(i + a) * pc + (j + b)] * ptr2[a * kc + b];

                        if (i < 1 & j < 1)py::print("suan c++", i, j, a, b, padded_ptr[(i + a) * pc + (j + b)], ptr2[a * kc + b]);
                        r = sum - y;
                        y = (r - sum) + y;
                        sum = r;
                    }
                ptr[i * c + j] = sum;

                //if(i<1&j<1)py::print("c++", i, j, ptr[i * c + j]);

            }
        /* for (int i = 0; i < kr; ++i)
             for (int d = 0; d < kc; ++d)
                 py::print("c++ kernal i,j:", i, d, ptr2[i * kc + d]);*/
    }
    else if (gpu == true)
    {
        int size[8];
        size8[0] = r; size8[2] = kr; size8[4] = hkr; size8[6] = pr;
        size8[1] = c; size8[3] = kc; size8[5] = hkc; size8[7] = pc;
        Convolve_compute(ptr1, ptr2, padded_ptr, ptr, size8);
    }
    return out;
}

py::array_t<double>  C_Convolve::get_paddle()
{
    py::array_t<double> padded = py::array_t<double>({ pr,pc });
    py::buffer_info buf = padded.request();
    buf.ptr = padded_ptr;
    return padded;
}