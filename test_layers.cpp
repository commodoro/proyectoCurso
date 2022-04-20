/* Ejemplo sin usar la clase red */

#include "./NNLib/NNLib.hpp"
#include "./data/iris.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
    // Definición
    NN::NormLayer A{4};
    NN::WGLayer B{&A, 8};
    NN::ReLuLayer C{&B};
    NN::WGLayer D{&C, 3};
    NN::SoftMaxLayer E{&D};

    // Configuración
    A.loadMeans("./data/means.csv");
    A.loadSD("./data/sd.csv");
    B.loadWeights("./data/w1.csv");
    B.loadBias("./data/w1.csv");
    D.loadWeights("./data/w2.csv");
    D.loadBias("./data/w2.csv");

    // Carga
    std::copy(data[0], data[0]+4, A.getMutInputBlock());
    
    // Ejecución
    A.compute();
    B.compute();
    C.compute();
    D.compute();
    E.compute();

    // Descarga
    float out[3];
    std::copy(E.getOutputBlock(), E.getOutputBlock()+3, out);

    // Print
    std::cout << "Entrada" << std::endl;
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << data[0][i] << ", ";
    }
    std::cout << "\b\b " << std::endl;
    std::cout << "Salida" << std::endl;
    for (size_t i = 0; i < 3; i++)
    {
        std::cout << out[i] << ", ";
    }
    std::cout << "\b\b " << std::endl;
    
    return 0;
}
