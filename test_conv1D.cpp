/* Ejemplo convolución 1D: Derivada */

#include "./NNLib/NNLib.hpp"
#include <iostream>
#include <math.h>

float kernel[3] = {-0.5,0,0.5};

int main(int argc, char const *argv[])
{
    NN::ConvLayer diff{11};

    std::cout << "f(x)=(x^2)/2\t[";
    for(int i = 0; i<11; ++i)
    {
        diff.getMutInputBlock()[i] = pow((float)i,2.0f)/2.0f;
        std::cout << diff.getInputBlock()[i] << ",\t";
    }
    std::cout << "\b\b]" << std::endl;

    diff.setKernel({{3},kernel});
    diff.compute();


    std::cout << "df(x)=x\t\t[";
    for(int i = 0; i<11; ++i)
    {
        std::cout << diff.getOutputBlock()[i] << ",\t";
    }
    std::cout << "\b\b]" << std::endl;

    return 0;
}
