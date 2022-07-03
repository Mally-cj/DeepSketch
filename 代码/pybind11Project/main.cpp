#include<pybind11/numpy.h>

#include"build/func.h"
 
//LibÎÄ¼þÂ·¾¶
namespace py = pybind11;

int add(int& a, int& b) {
	a = a + b;
	return a;
	
}

PYBIND11_MODULE(cspeed, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("add", &add);
	m.def("matAdd1", &matAdd1, "A function that adds two ");
	m.def("matmul", &matmul);
	m.def("C_SoftMax_compute", &C_SoftMax_compute );

	py::class_<C_Logistic>(m, "C_Logistic")
		.def(py::init<>())
		.def("compute", &C_Logistic::compute) 
		.def("getjacobi", &C_Logistic::getjacobi);


	py::class_<C_LogLoss>(m, "C_LogLoss")
		.def(py::init<>())
		.def("compute", &C_LogLoss::compute) 
		.def("getjacobi", &C_LogLoss::getjacobi)
		.def("init", &C_LogLoss::init);

	py::class_<C_CrossEntropyWithSoftMax>(m, "C_CrossEntropyWithSoftMax")
		.def(py::init<>())
		.def("compute", &C_CrossEntropyWithSoftMax::compute);

	py::class_<C_Convolve>(m, "C_Convolve")
		.def(py::init<>())
		.def("compute", &C_Convolve::compute)
		.def("get_paddle", &C_Convolve::get_paddle);

	
}


//py::class_<MyClass>(m, "C_Logistic")
//.def(py::init<>())
//.def("copy_matrix", &MyClass::getMatrix) // Makes a copy!
//.def("get_matrix", &MyClass::getMatrix, py::return_value_policy::reference_internal)
//.def("view_matrix", &MyClass::viewMatrix, py::return_value_policy::reference_internal);