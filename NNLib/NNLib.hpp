#include <memory>
#include <functional>
#include <cstdio>
#include "NNUtils.hpp"

namespace NN{

enum class OPCODE : uint8_t {
    OK
};

template<typename T = float>    
class GenericLayer
{
    private:
        /* Solo se pueden crear dinámicamente dentro de la clase Network*/
        friend class Network;
        void* operator new(size_t);
        void* operator new[](size_t);
        void operator delete(void*);
        void operator delete[](void*);
    protected:
        std::shared_ptr<T> _out;
        std::shared_ptr<T> _in;
        uint16_t _size_i, _size_o;
    public:
        GenericLayer() = delete;
        // GenericLayer(const GenericLayer&) = delete; Puede
        GenericLayer(const uint16_t &input_len, const uint16_t &output_len) : _size_i(input_len), _size_o(output_len)
        {
            _in = std::shared_ptr<T>{new T[input_len]};
            _out = std::shared_ptr<T>{new T[output_len]};
        }
        GenericLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block, const uint16_t &output_len) : _size_i(input_len), _size_o(output_len), _in(input_block)
        {
            _out = std::shared_ptr<T>{new T[output_len]};
        };
        GenericLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block) : GenericLayer(input_len, input_block, input_len) {};
        GenericLayer(const GenericLayer<T> * prev_layer, const uint16_t output_len) : _size_i(prev_layer->_size_o), _size_o(output_len), _in(prev_layer->_out)
        {
            _out = std::shared_ptr<T>{new T[output_len]};
        }
        ~GenericLayer() = default;

        virtual void compute() const {};

        T* getInputBlock() const {return _in.get();}
        T* getOutputBlock() const {return _out.get();}
        T* getMutInputBlock() {return _in.get();}

        uint16_t getInputSize() const {return _size_i;};
        uint16_t getOutputSize() const {return _size_o;};
};

template<typename T = float>
class LambdaLayer : public GenericLayer<T>
{
    private:
        friend class Network;
        void* operator new(size_t);
        void* operator new[](size_t);
        void operator delete(void*);
        void operator delete[](void*);
        std::function<void(T*, T*, uint16_t, uint16_t)> _app = [](T*, T*, uint16_t, uint16_t){};
    public:
        LambdaLayer() = delete;
        LambdaLayer(const uint16_t &input_len, const uint16_t &output_len) : GenericLayer<T>(input_len, output_len){}
        LambdaLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block, const uint16_t &output_len) : GenericLayer<T>(input_len, input_block, output_len){};
        LambdaLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block) :  GenericLayer<T>(input_len, input_block, input_len) {};
        LambdaLayer(const GenericLayer<T> * prev_layer, const uint16_t output_len) : GenericLayer<T>(prev_layer, output_len) {};
        void setApp(std::function<void(T*, T*, uint16_t, uint16_t)> app) {_app = app;}
        void compute() {
            _app(this->_in.get(), this->_out.get(), this->_size_i, this->_size_o);
        }
};

template<typename T = float>
class WGLayer final : public GenericLayer<T> 
{
    private:
        friend class Network;
        void* operator new(size_t);
        void* operator new[](size_t);
        void operator delete(void*);
        void operator delete[](void*);
        std::unique_ptr<T> _W;
        std::unique_ptr<T> _S;
    public:
        WGLayer() = delete;
        WGLayer(const uint16_t &input_len, const uint16_t &output_len) : GenericLayer<T>(input_len, output_len){
            this->_S = std::unique_ptr<T>{new T[this->_size_o]};
            this->_W = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        }
        WGLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block, const uint16_t &output_len) : GenericLayer<T>(input_len, input_block, output_len){
            this->_S = std::unique_ptr<T>{new T[this->_size_o]};
            this->_W = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        };
        WGLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) :  WGLayer<T>(layer_len, input_block, layer_len) {};
        WGLayer(const GenericLayer<T> * prev_layer, const uint16_t output_len) : GenericLayer<T>(prev_layer, output_len) {
            this->_S = std::unique_ptr<T>{new T[this->_size_o]};
            this->_W = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        };
        void compute() {
            for(uint_fast16_t i = 0; i<this->_size_o; ++i)
            {
                this->_out.get()[i] = this->_S.get()[i];
                for(uint_fast16_t j = 0; j < (this->_size_i); ++j)
                {
                    this->_out.get()[i] += this->_W.get()[i*this->_size_i+j]*this->_in.get()[j];
                }
            }
        }
        T* getWeights() const {return this->_W.get();}
        T* getMutWeights() {return this->_W.get();}
        T* getBias() const {return this->_S.get();}
        T* getMutBias() {return this->_S.get();}
        uint16_t getWCols() const {return this->_size_i;}
        uint16_t getWRows() const {return this->_size_o;}
        void loadWeights(FILE* fptr)
        {
            parseCSV(fptr, this->_W.get(), this->_size_i*this->_size_o);
        }
        void loadBias(FILE* fptr)
        {
            parseCSV(fptr, this->_S.get(), this->_size_o);
        }
        void loadWeights(const char* filename)
        {
            loadWeights(fopen(filename, "r"));
        }
        void loadBias(const char* filename)
        {
            loadBias(fopen(filename, "r"));
        }
};

template<typename T = float>
class ReLULayer final : public GenericLayer<T>
{
    private:
        friend class Network;
        void* operator new(size_t);
        void* operator new[](size_t);
        void operator delete(void*);
        void operator delete[](void*);
    public:
        ReLULayer() = delete;
        ReLULayer(const uint16_t &layer_len) : GenericLayer<T>(layer_len, layer_len){};
        ReLULayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) : GenericLayer<T>(layer_len, input_block, layer_len){};
        ReLULayer(const GenericLayer<T> * prev_layer) : GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {};
        uint16_t getLayerLen() const {return this->_size_i;}
        void compute()
        {
            for (size_t i = 0; i < this->_size_i; i++)
            {
                this->_out.get()[i] = this->_in.get()[i] > 0? this->_in.get()[i] : 0;
            }
        }
};

template<typename T = float>
class NormLayer final : public GenericLayer<T>
{
    private:
        friend class Network;
        void* operator new(size_t);
        void* operator new[](size_t);
        void operator delete(void*);
        void operator delete[](void*);
        std::unique_ptr<T> _M;
        std::unique_ptr<T> _S;
    public:
        NormLayer() = delete;
        NormLayer(const uint16_t &layer_len) : GenericLayer<T>(layer_len, layer_len){
            this->_M = std::unique_ptr<T>{new T[layer_len]};
            this->_S = std::unique_ptr<T>{new T[layer_len]};
        };
        NormLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) : GenericLayer<T>(layer_len, input_block, layer_len){
            this->_M = std::unique_ptr<T>{new T[layer_len]};
            this->_S = std::unique_ptr<T>{new T[layer_len]};
        };
        NormLayer(const GenericLayer<T> * prev_layer) : GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {
            this->_M = std::unique_ptr<T>{new T[this->_size_o]};
            this->_S = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        };
        void compute()
        {
            for (size_t i = 0; i < this->_size_i; i++)
            {
                if (this->_S.get()[i] == 0)
                {
                    // Aquí tendría que poner un flag de erorr.
                    continue;
                }
                this->_out.get()[i] = (this->_in.get()[i]-this->_M.get()[i])/this->_S.get()[i];
            }
        }
        T* getMeans() const {return this->_M.get();}
        T* getMutMeans() {return this->_M.get();}
        T* getSD() const {return this->_S.get();}
        T* getMutSD() {return this->_S.get();}
        uint16_t getLayerLen() const {return this->_size_i;}
        void loadMeans(FILE* fptr)
        {
            parseCSV(fptr, this->_M.get(), this->_size_i);
        }
        void loadSD(FILE* fptr)
        {
            parseCSV(fptr, this->_S.get(), this->_size_o);
        }
        void loadMeans(const char* filename)
        {
            loadMeans(fopen(filename, "r"));
        }
        void loadSD(const char* filename)
        {
            loadSD(fopen(filename, "r"));
        }
};


class Network
{
    // A medio
};




}