#include "./NNLib/NNLib.hpp"
#include "./data/iris.hpp"
#include <iostream>


int main(int argc, char const *argv[])
{
    float out[3];
    int muestra;
    auto net = NN::loadNet<float>("./data/nn1.toml");

    std::cout << "Seleccionar muestra [0-149]: ";
    std::cin >> muestra;

    net.copy2input(data[muestra]);
    net.init();
    net.compute();
    net.copyout(out);

    std::cout << "Entrada: ";
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << data[muestra][i] << ", ";
    }
    std::cout << "\b\b " << std::endl;
    
    std::cout << "Salida: ";
    for (size_t i = 0; i < 3; i++)
    {
        std::cout << out[i] << ", ";
    }
    std::cout << "\b\b " << std::endl;

    return 0;
}
