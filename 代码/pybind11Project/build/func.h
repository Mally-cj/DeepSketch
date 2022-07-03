#pragma once
#include<pybind11/numpy.h>
#include <cmath>
#include<stdio.h>
#include<cstring>
#include"../../DLLLp//Header.cuh"  //头文件路径
#pragma comment (lib, "E:/software/Anaconda3/envs/pytorch/C++project/0601new3/Py/from_other/DLLLp.lib")   

namespace py = pybind11;

//func的
class C_Logistic {
public:
    double* value_ptr;  //value的值的指针
    int value_len;      //value值的大小
    py::array_t<double> compute(py::array_t<double>& x, int gpu);
    py::array_t<double> getjacobi();
};


class C_Convolve {
public:
    bool usecpmpute;
    int size8[8];
    int r,c;
    int kr, kc;
    int hkr, hkc;
    int pr, pc;
   
    double* padded_ptr;
    py::array_t<double> compute(py::array_t<double>& data, py::array_t<double>& kernel,bool gpu);
    py::array_t<double> get_paddle();
    py::array_t<double> compute2(py::array_t<double>& data, py::array_t<double>& kernel, bool gpu);

};


//funcBase的
py::array_t<double> matAdd(py::array_t<double>& arr_a, py::array_t<double>& arr_b);
py::array_t<double> matAdd1(py::array_t<double>& arr_a, py::array_t<double>& arr_b);

py::array_t<double> matmul(const py::array_t<double>& aa, const py::array_t<double>& bb, bool gpu);
py::array_t<double> change2(py::array_t<double>& input);

//losfunc
class C_LogLoss {
public:
    void init();
    py::array_t<double> compute(const py::array_t<double>& x, bool gpu);
    py::array_t<double> getjacobi(const py::array_t<double>& x,bool gpu);
};

py::array_t<double> C_SoftMax_compute(const py::array_t<double>& x, bool gpu);

class C_CrossEntropyWithSoftMax {
public:
    double* prob_ptr;
    py::array_t<double> compute(const py::array_t<double>& x1, const py::array_t<double>& x2, bool gpu);
};

//