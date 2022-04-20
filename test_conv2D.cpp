/* Test Convolución 2D */

#include "./NNLib/NNLib.hpp"
#include <iostream>

float kernel[3][3] = {
    {0.00,0.25,0.00},
    {0.25,0.00,0.25},
    {0.00,0.25,0.00},
};

float matrix[10][10] = {
    {1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,0,0,1,1,1,1},
    {1,1,1,0,0,0,0,1,1,1},
    {1,1,0,0,0,0,0,0,1,1},
    {1,1,0,0,0,0,0,0,1,1},
    {1,1,1,0,0,0,0,1,1,1},
    {1,1,1,1,0,0,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1}
};

void printMatrix(float* mat, NN::dim_t dim)
{
    for (size_t i = 0; i < dim.rows; i++)
    {
        for (size_t j = 0; j < dim.rows; j++)
        {
            std::cout << mat[i*dim.rows+j] << "\t";  
        }
        std::cout << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    // Definición
    NN::ConvLayer conv({10,10});

    // Configuración
    conv.setKernel({{3,3},kernel[0]});
    conv.setPadding(NN::ConvPadding::SAME);

    // Entrada
    std::copy(matrix[0], matrix[0]+100, conv.getMutInputBlock());

    // Ejecución
    conv.compute();

    // Print
    std::cout << "Kernel:" << std::endl;
    printMatrix(kernel[0],{3,3});

    std::cout << "Entrada:" << std::endl;
    printMatrix(matrix[0], {10,10});

    std::cout << "Salida con padding SAME (Extiende bordes):" << std::endl;
    printMatrix(conv.getOutputBlock(), {10,10});

    conv.setPadding(NN::ConvPadding::VALID);
    conv.compute();

    std::cout << "Salida con padding VALID (Rellena de ceros):" << std::endl;
    printMatrix(conv.getOutputBlock(), {10,10});
    
    return 0;
}
