

#include "mylib.h"


 
int myfunc(numpyArray<double> array1, numpyArray<double> array2)
{
    Ndarray<double,3> a(array1);
    Ndarray<double,3> b(array2);

    double sum=0.0;

    for (int i = 0; i < a.getShape(0); i++)
    {
        for (int j = 0; j < a.getShape(1); j++)
        {
            for (int k = 0; k < a.getShape(2); k++)
            {
                a[i][j][k] = 2.0 * b[i][j][k];
                sum += a[i][j][k];
           }
        }
    }
    return sum;    
}

