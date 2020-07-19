

#include <iostream>
#include <string>
#include <vector>
#include "ndarray.h"

using namespace std;

extern "C"
{
    int myfunc(numpyArray<double> array1, numpyArray<double> array2);
}
