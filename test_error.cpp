/* Test de manejo de erorres */

#include "./NNLib/NNLib.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
    NN::Net net(4);
    net.addWGLayer(8, "./data/w1.csv", "./data/b1.csv");
    net.addSigmoidLayer();
    net.addWGLayer(3, "./ddata/w2.csv", "./data/b2.csv"); 
    // error--------------^
    net.addSoftMaxLayer();

    // Descomentar el nivel de excepción deseado. 
    // net.except_level(NN::EXCEPLEVEL::ZERO);         // Sin aviso
    // net.except_level(NN::EXCEPLEVEL::CERR);         // Salida por cerr
    net.except_level(NN::EXCEPLEVEL::THROW_ALL);       // Lanza iterrupción, por defecto

    net.init();

    return 0;
}
