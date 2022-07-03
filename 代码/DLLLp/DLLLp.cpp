// DLLLp.cpp : 定义 DLL 的导出函数。
//

#include "pch.h"
#include "framework.h"
#include "DLLLp.h"


// 这是导出变量的一个示例
DLLLP_API int nDLLLp=0;

// 这是导出函数的一个示例。
DLLLP_API int fnDLLLp(void)
{
    return 0;
}

// 这是已导出类的构造函数。
CDLLLp::CDLLLp()
{
    return;
}
