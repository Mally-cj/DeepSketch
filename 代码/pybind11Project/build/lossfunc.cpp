#include"func.h"
void C_LogLoss::init()
{
    py::print("create C_Logloss\n");
}
py::array_t<double> C_LogLoss::compute(const py::array_t<double>& x, bool gpu)
{
    //self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    py::buffer_info bufx = x.request();
    double* ptrx = (double*)bufx.ptr;

    auto out = py::array_t<double>({ bufx.shape[0], bufx.shape[1] });
    py::buffer_info buf = out.request();
    double* ptr = (double*)buf.ptr;

    int size = bufx.shape[0] * bufx.shape[1];
   
    if (gpu == false)
    {
        for (int i = 0; i < size; ++i)
        {
            double tem = ptrx[i] < -100.0 ? 100 : ptrx[i]*(-1.0);
            ptr[i] = log(1 + exp(tem));
        }
    }
    else
    {
        Logloss_compute(ptrx, ptr, size);
    }
    return out;
}

py::array_t<double> C_LogLoss::getjacobi(const py::array_t<double>& x,bool gpu)
{ 
    /*
  python code:
       diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
       return np.diag(diag.ravel())
  */
    py::buffer_info bufx = x.request();
    double* ptrx = (double*)bufx.ptr;

    int size = bufx.shape[0] * bufx.shape[1];
    auto out = py::array_t<double>({ size, size });
    py::buffer_info buf = out.request();
    double* ptr = (double*)buf.ptr;
    memset(ptr, 0, sizeof(ptr));

    if (gpu == false)
    {
        int index = 0;
        for (int r = 0; r < size; ++r)
        {
            double tem = ptrx[r] > 100.0 ? 100 : ptrx[r];
            ptr[index] = -1.0 / (1 + exp(tem));;
            index += (size + r);
        }
    }
    else
    {
        double *y= (double*)malloc(sizeof(double) *size);
        for (int i = 0; i < size; ++i)y[i] = ptrx[i];
        Logloss_getjacobi(y, size);
        int index = 0;
        for (int r = 0; r < size; ++r)
        {
            ptr[index] = y[r];
            index += (size + r);
        }
        free(y);
        y = NULL;
    }
    return out;
}


py::array_t<double> C_SoftMax_compute(const py::array_t<double>& x, bool gpu)
{
    /*
     python code:
     a[a > 1e2] = 1e2 
     ep = np.power(np.e, a)
     return ep / np.sum(ep)
    */

    py::buffer_info bufx = x.request();
    double* ptrx = (double*)bufx.ptr;
    int size = bufx.shape[0] * bufx.shape[1];

    py::array_t<double> out = py::array_t<double>({ bufx.shape[0], bufx.shape[1] });
    py::buffer_info buf = out.request();
    double* ptr = (double*)buf.ptr;
    

    if (gpu == true)
    {
        int block_num = (size + 256 - 1) / 256;  
        //这里是默认THREAD_NUM=256，原式是block_num = (DATA_SIZE + THREAD_NUM - 1) / THREAD_NUM;
        double* sum_tmp = (double*)malloc(sizeof(double) * block_num);
        Softmax_compute(ptrx, sum_tmp, ptr, size, block_num);
    }
    else if(gpu==false)
    {
        double he = 0;
        for (int i = 0; i < size; ++i)
        {
            ptr[i] = ptrx[i] > 100 ? 100 : ptrx[i];
            ptr[i] = exp(ptr[i]);
            he += ptr[i];
        }
        for (int i = 0; i < size; ++i)ptr[i] /= he;
    }
    return out;
}

py::array_t<double> C_CrossEntropyWithSoftMax::compute(const py::array_t<double>& x1, const py::array_t<double>& x2, bool gpu)
{
    /*
      prob = ope.SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))
    */

    py::buffer_info bufx1 = x1.request();
    double* ptrx1 = (double*)bufx1.ptr;
    int size = bufx1.shape[0] * bufx1.shape[1];

    py::buffer_info bufx2 = x2.request();
    double* ptrx2 = (double*)bufx2.ptr;

    prob_ptr = (double*)malloc(sizeof(double) * size);;
    
    py::array_t<double> out = py::array_t<double>({1,1});
    py::buffer_info buf = out.request();
    double* ptr = (double*)buf.ptr;

    if (gpu == false)
    {
        double he = 0;
        for (int i = 0; i < size; ++i)
        {
            prob_ptr[i] = ptrx1[i] > 100 ? 100 : ptrx1[i];
            prob_ptr[i] = exp(prob_ptr[i]);
            he += prob_ptr[i];
        }
        double sum = 0.0;
        double tem;
        for (int i = 0; i < size; ++i)
        {
            prob_ptr[i] /= he;
            sum += log(prob_ptr[i]+ (1e-10)) * ptrx2[i]*(-1);
        }
        ptr[0] = sum;
    }
    else if (gpu == true)
    {
        int block_num = (size + 256 - 1) / 256;
        //这里是默认THREAD_NUM=256，原式是block_num = (DATA_SIZE + THREAD_NUM - 1) / THREAD_NUM;
        double* sum_tmp = (double*)malloc(sizeof(double) * block_num);
        Softmax_compute(ptrx1, sum_tmp, prob_ptr, size, block_num);
        ptr[0]=CrossEntropyWithSoftMax_compute(prob_ptr, ptrx2, sum_tmp, size, block_num);
        free(sum_tmp);
        sum_tmp = NULL;
    }
    return out;
}

