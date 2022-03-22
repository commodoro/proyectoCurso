#include <iostream>
#include <memory>
#include "NNLib/NNLib.hpp"
#include <math.h>
#include <vector>

int main(int argc, char const *argv[])
{
    std::shared_ptr<float> ptrEntrada{new float[6]};
    NN::WGLayer A{6,ptrEntrada,6};
    NN::ReLULayer B{&A};
    NN::NormLayer C{&B};

    for (size_t i = 0; i < 6; i++)
    {
        A.getMutBias()[i] = 0;
        A.getMutInputBlock()[i] = 1;
    }
    
    A.loadWeights("./example.csv");
    C.loadMeans("./means.csv");
    C.loadSD("./sd.csv");
    A.compute();
    B.compute();
    C.compute();

    for (size_t i = 0; i < 6; i++)
    {
        std::cout << 1 << " >\t";
        std::cout << A.getOutputBlock()[i] << " >\t";
        std::cout << B.getOutputBlock()[i] << " >\t";
        std::cout << C.getOutputBlock()[i];
        std::cout << std::endl;
    }
    
    return 0;
}
