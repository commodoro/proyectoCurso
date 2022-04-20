#ifndef __NN_NNUTILS__
#define __NN_NNUTILS__

#include <type_traits>
#include <cinttypes>
#include <fstream>
#include <iterator>

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

namespace NN{

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