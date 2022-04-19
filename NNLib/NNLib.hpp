#include <memory>
#include <functional>
#include <cstdio>
#include "NNUtils.hpp"
#include <math.h>
#include <type_traits>
#include <vector>

namespace NN{

enum class OPCODE : uint8_t {
    OK
};

template<typename T = float> class Net;

template<typename T = float>    
class GenericLayer
{
    static_assert(std::is_floating_point<T>::value, "A Layer class can only be instantiated with floating point types.");
    private:
        /* Solo se pueden crear dinámicamente dentro de la clase Net*/
        friend class Net<T>;
    protected:
        std::shared_ptr<T> _out;
        std::shared_ptr<T> _in;
        uint16_t _size_i, _size_o;
    public:
        GenericLayer() = default;
        // GenericLayer(const GenericLayer& other)
        // {
        //     _in = other._in;
        //     _out = other._out;
        //     _size_i = other._size_i;
        //     _size_o = other._size_o;
        // };
        GenericLayer(const uint16_t &input_len, const uint16_t &output_len) : _size_i(input_len), _size_o(output_len)
        {
            _in = std::shared_ptr<T>{new T[input_len]};
            _out = std::shared_ptr<T>{new T[output_len]};
        };
        GenericLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block, const uint16_t &output_len) : _size_i(input_len), _size_o(output_len), _in(input_block)
        {
            _out = std::shared_ptr<T>{new T[output_len]};
        };
        GenericLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block) : GenericLayer(input_len, input_block, input_len) {};
        GenericLayer(const GenericLayer<T> * prev_layer, const uint16_t output_len) : _size_i(prev_layer->_size_o), _size_o(output_len), _in(prev_layer->_out)
        {
            _out = std::shared_ptr<T>{new T[output_len]};
        };
        ~GenericLayer() = default;

        virtual void compute() {};

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
        friend class Net<T>;
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
        friend class Net<T>;
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
        friend class Net<T>;
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
        friend class Net<T>;
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
        void compute() override
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

template<typename T = float>
class SoftMaxLayer final : public GenericLayer<T>
{
    private:
        friend class Net<T>;
    public:
        SoftMaxLayer() = delete;
        SoftMaxLayer(const uint16_t &layer_len) : GenericLayer<T>(layer_len, layer_len){};
        SoftMaxLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) : GenericLayer<T>(layer_len, input_block, layer_len){};
        SoftMaxLayer(const GenericLayer<T> * prev_layer) : GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {};
        uint16_t getLayerLen() const {return this->_size_i;}
        
        void compute()
        {
            T eacc = 0;
            for (size_t i = 0; i < this->_size_i; i++)
            {
                this->_out.get()[i] = exp(this->_in.get()[i]);
                eacc += this->_out.get()[i];
            }
            if (eacc == 0)
                return; // Debería dar un error.
            for (size_t i = 0; i < this->_size_i; i++)
            {
                this->_out.get()[i] = this->_out.get()[i]/eacc;
            }
        }
};

template<typename T>
class Net
{
    static_assert(std::is_floating_point<T>::value, "A Net class can only be instantiated with floating point types.");
    private:
        uint16_t _input_size, _output_size;
        std::vector<std::unique_ptr<GenericLayer<T>>> _layer_list;
        std::shared_ptr<T> _in;
        std::shared_ptr<T> _out;
    public:
        Net() = delete;
        Net(const uint16_t &input_len) : _input_size(input_len)
        {
            _in = std::shared_ptr<T>{new T[input_len]};
        }

        // Lambda
        void addLambdaLayer(const uint16_t &output_len, std::function<void(T*, T*, uint16_t, uint16_t)> app)
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new LambdaLayer<T>(_input_size, _in, output_len));
            }
            else
            {
                _layer_list.emplace_back(new LambdaLayer<T>(_layer_list.back().get(), output_len));
            }
            auto lamptr = dynamic_cast<LambdaLayer<T>*>(_layer_list.back().get());
            lamptr->setApp(app);
        }
        
        // WG
        void addWGLayer(const uint16_t &output_len, T* w_first, T* s_first)
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new WGLayer<T>(_input_size, _in, output_len));
            }
            else
            {
                _layer_list.emplace_back(new WGLayer<T>(_layer_list.back().get(), output_len));
            }
            auto wgptr = dynamic_cast<WGLayer<T>*>(_layer_list.back().get());
            std::copy(w_first,
                      w_first + (wgptr->getInputSize()*wgptr->getOutputSize()),
                      wgptr->getMutWeights());      
            std::copy(s_first, s_first+wgptr->getOutputSize(), wgptr->getMutBias());     
        }
        void addWGLayer(const uint16_t &output_len, const char* file_w, const char* file_s)
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new WGLayer<T>(_input_size, _in, output_len));
            }
            else
            {
                _layer_list.emplace_back(new WGLayer<T>(_layer_list.back().get(), output_len));
            }
            auto nptr = dynamic_cast<WGLayer<T>*>(_layer_list.back().get());
            nptr->loadWeights(file_w);
            nptr->loadBias(file_s);
        }

        // Normalize
        void addNormLayer(T* m_first, T* sd_first)
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new NormLayer<T>(_input_size, _in));
            }
            else
            {
                _layer_list.emplace_back(new NormLayer<T>(_layer_list.back().get()));
            }
            auto nptr = dynamic_cast<NormLayer<T>*>(_layer_list.back().get());
            std::copy(m_first, m_first+nptr->getInputSize(), nptr->getMutMeans());      
            std::copy(sd_first, sd_first+nptr->getInputSize(), nptr->getMutSD());      
        }
        void addNormLayer(const char* file_m, const char* file_sd)
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new NormLayer<T>(_input_size, _in));
            }
            else
            {
                _layer_list.emplace_back(new NormLayer<T>(_layer_list.back().get()));
            }
            auto nptr = dynamic_cast<NormLayer<T>*>(_layer_list.back().get());
            nptr->loadMeans(file_m);
            nptr->loadSD(file_sd);
        }

        // Relu
        void addReLuLayer()
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new ReLULayer<T>(_input_size, _in));
            }
            else
            {
                _layer_list.emplace_back(new ReLULayer<T>(_layer_list.back().get()));
            }
        }

        // SoftMax
        void addSoftMaxLayer()
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new SoftMaxLayer<T>(_input_size, _in));
            }
            else
            {
                _layer_list.emplace_back(new SoftMaxLayer<T>(_layer_list.back().get()));
            }
        }


        // Computar
        void compute()
        {
            for(auto &layer: this->_layer_list)
                layer->compute();
        }
        void operator()()
        {
            this->compute();
        }

        // Inicializar
        void init()
        {
            _out = std::shared_ptr<T>{_layer_list.back()->_out};
            _output_size = _layer_list.back()->_size_o;
        }

        void copy2input(const T* first)
        {
            std::copy(first, first+_input_size, _in.get());
        }

        // Block memory
        T* getInput() {return _in.get();}
        T* getInput() const {return _in.get();}
        T* getOutput() const {return _out.get();}

        // Lengths
        uint16_t getInputSize() const {return _input_size;};
        uint16_t getOutputSize() const {return _output_size;};

};

}