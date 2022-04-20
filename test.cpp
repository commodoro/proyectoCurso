#include <iostream>
#include "NNLib/NNLib.hpp"
#include "data/iris.hpp"

int main(int argc, char const *argv[])
{
    NN::Net net(4);
    net.addNormLayer("./data/means.csv", "./data/sd.csv");
    net.addWGLayer(8, "./data/w1.csv","./data/b1.csv");
    net.addReLuLayer();
    net.addWGLayer(3, "./data/w2.csv","./data/b2.csv");
    net.addSoftMaxLayer();
    
    net.init();

    float res[150][3];

    for (size_t i = 0; i < 150; i++)
    {
        net.copy2input(data[i]); // Copia los datos a la entrada.
        net(); // Igual que net.compute();
        net.copyout(res[i]); // Copia los datos de salida a un array.
    }

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

    return 0;
}
