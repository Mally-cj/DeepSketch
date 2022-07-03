cmake_minimum_required(VERSION 2.8.12)  #这个是cmake编译后的工程名，可以自己定义
project(cspeed)
add_subdirectory(pybind11)
pybind11_add_module(cspeed main.cpp)  #括号内第一个为工程名，和上面一样，后面包含我们需要的所有文件

