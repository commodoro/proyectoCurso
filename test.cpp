/* Showcase básico */

#include <iostream>
#include "NNLib/NNLib.hpp"
#include "data/iris.hpp" // data[150][4] y expected[150][3]

float res[150][3]; // Donde guardamos la salida.

void test()
{
    bool cond[150];
    std::cout << "[Muestra]\t[Resultados calculados]\t\t[Resultados esperados]\t[Mayor %]" << std::endl;
    for (int i = 0; i<150; ++i)
    {
        int mayor_r = res[i][0] > res[i][1]? (res[i][0] < res[i][2])*2  : (res[i][1] < res[i][2]) + 1;
        int mayor_e = expected[i][0] > expected[i][1]? (expected[i][0] < expected[i][2])*2  : (expected[i][1] < expected[i][2]) + 1;
        cond[i] = mayor_e == mayor_r;
        printf("[%03d]     \t[%.3f, %.3f, %.3f]\t\t[%.3f, %.3f, %.3f]\t[%d,%d]\n", i,
        res[i][0], res[i][1], res[i][2], expected[i][0], expected[i][1], expected[i][2], mayor_r, mayor_e);
    }
    std::cout << "Aciertos: " << std::count_if(cond,cond+150, [](bool x) {return x;}) << "/150" << std::endl;
}

int main(int argc, char const *argv[])
{
    NN::Net net(4);                                         // Creamos una red con una entrada de 4 elementos.
    net.addNormLayer("./data/means.csv", "./data/sd.csv");  // Capa de normalización.
    net.addWGLayer(8, "./data/w1.csv","./data/b1.csv");     // Capa de pesos + sesgo
    net.addReLuLayer();                                     // Capa Relu
    net.addWGLayer(3, "./data/w2.csv","./data/b2.csv");
    net.addSoftMaxLayer();                                  // Capa SoftMax
    
    net.init(); // Inicializamos. Esto enlaza la salida de la última capa con la capa de salida de la red.

    for (size_t i = 0; i < 150; i++)
    {
        net.copy2input(data[i]);    // Copia los datos a la entrada.
        net();                      // Igual que net.compute();
        net.copyout(res[i]);        // Copia los datos de salida a un array.
    }

    test(); // Muestra los resultados

    return 0;
}
