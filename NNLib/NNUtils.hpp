#ifndef __NN_NNUTILS__
#define __NN_NNUTILS__

#include <type_traits>
#include <cinttypes>
#include <fstream>
#include <iterator>
#include <string>
#include "./libs/toml11/toml.hpp"
#include <exception>
#include <sstream>


#define NN_BUFF_SIZE_FS_REISERFS 4096
#define NN_BUFF_SIZE_FS_EXT3_1K 1024
#define NN_BUFF_SIZE_FS_EXT3_2K 2048
#define NN_BUFF_SIZE_FS_EXT3_4K 4096
#define NN_BUFF_SIZE_FS_EXT4 4096

#ifndef NN_PARSECSV_BUFF_USER_SIZE
#define NN_PARSECSV_BUFF_SIZE 1024 // Default
#elif
#define NN_PARSECSV_BUFF_SIZE NN_PARSECSV_BUFF_USER_SIZE // Default
#endif

#define NN_NO_WARNINGS // Comment to enable warns

namespace NN{

enum class OPCODE : uint16_t {
    OK,
    WARN_0 = 20, //  // Sobran datos en el fichero que han sido descartados
    BUILD_ERROR_0 = 100, // Tamaño de entrada/salida menor que 1.
    BUILD_ERROR_1, // Error con la capa sobre la que se pretende construir.
    BUILD_ERROR_2, // Dimensiones incoherentes.
    OP_ERROR_0 = 200, // Operación no permitida porque el estado actual no es OK.
    OP_ERROR_1, // La función lambda no está asignada
    OP_ERROR_2, // División por cero en la operación.
    CONF_ERROR_0 = 300, // Sin kernel.
    CONF_ERROR_1, // Dimensiones del kernel inconsistentes.
    CONF_ERROR_2, // El puntero del kernel no apunta a ningún bloque de memoria.
    PARS_ERROR_0 = 400, // Fichero no encontrado
    PARS_ERROR_1,  // Error al abrir fichero.
    PARS_ERROR_2,   // Fichero vacío
    PARS_ERROR_3    // Faltan datos
};

enum class EXCEPLEVEL : char {
    ZERO = 0,
    CERR,
    THROW_ALL
};

std::ostream& operator<<(std::ostream& os, const OPCODE &code)
{
    os << "[OPCODE]: ";
    switch (code)
    {
    case OPCODE::OK:
        os << "Normal operation. No error.";
        break;
    case OPCODE::WARN_0:
        os << "There is excess data in the file that has been discarded.";
        break;
    case OPCODE::BUILD_ERROR_0:
        os << "Input/output size less than 1.";
        break;
    case OPCODE::BUILD_ERROR_1:
        os << "Error with the layer on which it is intended to build.";
        break;
    case OPCODE::BUILD_ERROR_2:
        os << "Inconsistent dimensions.";
        break;
    case OPCODE::OP_ERROR_0:
        os << "Operation not allowed because the current state is not OK.";
        break;
    case OPCODE::OP_ERROR_1:
        os << "Lambda function is not assigned.";
        break;
    case OPCODE::OP_ERROR_2:
        os << "Division by zero in the operation.";
        break;
    case OPCODE::CONF_ERROR_0:
        os << "Kernel not configured.";
        break;
    case OPCODE::CONF_ERROR_1:
        os << "Inconsistent kernel dimensions.";
        break;
    case OPCODE::CONF_ERROR_2:
        os << "The kernel pointer does not point to any memory blocks.";
        break;
    case OPCODE::PARS_ERROR_0:
        os << "File not found;";
        break;
    case OPCODE::PARS_ERROR_1:
        os << "Error opening file.";
        break;
    case OPCODE::PARS_ERROR_2:
        os << "Empty file.";
        break;
    case OPCODE::PARS_ERROR_3:
        os << "Missing data in the file.";
        break;
    default:
        os << "Unknow code.";
        break;
    }
    return os;
}

struct NetError : public std::exception
{
    NetError() = delete;
    NetError(OPCODE code) : std::exception()
    {
        std::ostringstream ss_msg{std::ostringstream::ate};
        ss_msg.str("Neural network runtime error. ");
        ss_msg << code;
        msg = ss_msg.str();
    }
    NetError(OPCODE code, int layer, const char* id) : std::exception()
    {
        std::ostringstream ss_msg{std::ostringstream::ate};
        ss_msg.str("Neural network runtime error. In layer: ");
        ss_msg << layer << " [type:" << id << "]. " << code;
        msg = ss_msg.str();
    }
    std::string msg;
    const char* what() const noexcept {
        return msg.c_str();
    }
};

struct LoadError : public std::exception
{
    LoadError() = delete;
    LoadError(std::string msg) : msg(msg), std::exception() {};
    std::string msg;
    const char* what() const noexcept {
        return msg.c_str();
    }
};


template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, int>::type parseCSV(FILE *pFile, T *dest, uint_fast16_t dest_len)
{
    char num[40];
    char buff[NN_PARSECSV_BUFF_SIZE];

    if(pFile==NULL)
        return -1;

    if(ferror(pFile))
        return -2;

    if(feof(pFile))
        return -3;

    fseek(pFile, 0L, SEEK_END);
    size_t f_size = ftell(pFile);
    rewind(pFile);
    
    int n_tokens = f_size/NN_PARSECSV_BUFF_SIZE;

    size_t i ;
    size_t j = 0;
    size_t n = 0;

    while(n_tokens >= 0)
    {
        size_t r = fread(buff, 1, NN_PARSECSV_BUFF_SIZE, pFile);
        if(!(n_tokens--) && r < NN_PARSECSV_BUFF_SIZE)
            buff[r++] = '\0';
        i = 0;
        while(i<r)
        {
            if (n == dest_len)
                return n;

            char c = buff[i++];

            if (c >= '0' && c <= '9' || c == '.' || c == '-' || c == '+' || c == 'e')
            {
                num[j++] = c;
                continue;
            }

            if ((c == ',') || (c <= '\x20'))
            {
                num[j++] = '\0';
                dest[n++] = atof(num);
                j = 0;
            } else {
                return -4;
            }
        }
    };

    if (n==dest_len)
        return 0;
    
    return n;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, int>::type parseCSV(FILE *pFile, T *dest, uint_fast16_t dest_len)
{
    char num[30];
    char buff[NN_PARSECSV_BUFF_SIZE];

    if(pFile==NULL)
        return -1;

    if(ferror(pFile))
        return -2;

    if(feof(pFile))
        return -3;

    fseek(pFile, 0L, SEEK_END);
    size_t f_size = ftell(pFile);
    rewind(pFile);
    
    int n_tokens = f_size/NN_PARSECSV_BUFF_SIZE;

    size_t i ;
    size_t j = 0;
    size_t n = 0;

    while(n_tokens >= 0)
    {
        size_t r = fread(buff, 1, NN_PARSECSV_BUFF_SIZE, pFile);
        if(!(n_tokens--) && r < NN_PARSECSV_BUFF_SIZE)
            buff[r++] = '\0';
        i = 0;
        while(i<r)
        {
            if (n == dest_len)
                return n;

            char c = buff[i++];

            if (c >= '0' && c <= '9' || c == '-' || c == '+')
            {
                num[j++] = c;
                continue;
            }

            if ((c == ',') || (c <= '\x20'))
            {
                num[j++] = '\0';
                dest[n++] = atoi(num);
                j = 0;
            } else {
                return -4;
            }
        }
    };

    if (n==dest_len)
        return 0;
    
    return n;
}

typedef struct Dimensions
{
    uint16_t rows, cols;
    Dimensions(uint16_t c, uint16_t r=1) : rows(r), cols(c) {};
} dim_t;

template<typename T = float>
struct ConvKernel
{
    private:
        uint16_t _size;
        dim_t _dim;
    public:
        T* data;
        ConvKernel() : _dim(0,0), _size(0) {};
        ConvKernel(dim_t dim) : _dim(dim), _size(dim.cols*dim.rows){};
        ConvKernel(dim_t dim, T* data_ptr) : _dim(dim), _size(dim.cols*dim.rows), data(data_ptr) {};
        uint16_t rows() const {return _dim.rows;}
        uint16_t cols() const {return _dim.cols;}
        uint16_t size() const {return _size;}
};

enum class ConvPadding : char
{
    VALID, SAME
};

}

#endif