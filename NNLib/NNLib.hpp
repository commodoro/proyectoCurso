#ifndef __NN_NNLIB__
#define __NN_NNLIB__

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
        static const char _id[];
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
        virtual const char* id() const {return _id;}

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
        static const char _id[];
    public:
        LambdaLayer() = delete;
        LambdaLayer(const uint16_t &input_len, const uint16_t &output_len) : GenericLayer<T>(input_len, output_len){}
        LambdaLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block, const uint16_t &output_len) : GenericLayer<T>(input_len, input_block, output_len){};
        LambdaLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block) :  GenericLayer<T>(input_len, input_block, input_len) {};
        LambdaLayer(const GenericLayer<T> * prev_layer, const uint16_t output_len) : GenericLayer<T>(prev_layer, output_len) {};
        void setApp(std::function<void(T*, T*, uint16_t, uint16_t)> app) {_app = app;}
        void compute() override
        {
            _app(this->_in.get(), this->_out.get(), this->_size_i, this->_size_o);
        }
        const char* id() const override {return this->_id;}
};

template<typename T = float>
class WGLayer final : public GenericLayer<T> 
{
    private:
        friend class Net<T>;
        static const char _id[];
        std::unique_ptr<T> _W;
        std::unique_ptr<T> _B;
    public:
        WGLayer() = delete;
        WGLayer(const uint16_t &input_len, const uint16_t &output_len) : GenericLayer<T>(input_len, output_len){
            this->_B = std::unique_ptr<T>{new T[this->_size_o]};
            this->_W = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        }
        WGLayer(const uint16_t &input_len, const std::shared_ptr<T> &input_block, const uint16_t &output_len) : GenericLayer<T>(input_len, input_block, output_len){
            this->_B = std::unique_ptr<T>{new T[this->_size_o]};
            this->_W = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        };
        WGLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) :  WGLayer<T>(layer_len, input_block, layer_len) {};
        WGLayer(const GenericLayer<T> * prev_layer, const uint16_t output_len) : GenericLayer<T>(prev_layer, output_len) {
            this->_B = std::unique_ptr<T>{new T[this->_size_o]};
            this->_W = std::unique_ptr<T>{new T[this->_size_i*this->_size_o]};
        };
        void compute() override
        {
            for(uint_fast16_t i = 0; i<this->_size_o; ++i)
            {
                this->_out.get()[i] = this->_B.get()[i];
                for(uint_fast16_t j = 0; j < (this->_size_i); ++j)
                {
                    this->_out.get()[i] += this->_W.get()[i*this->_size_i+j]*this->_in.get()[j];
                }
            }
        }
        T* getWeights() const {return this->_W.get();}
        T* getMutWeights() {return this->_W.get();}
        T* getBias() const {return this->_B.get();}
        T* getMutBias() {return this->_B.get();}
        uint16_t getWCols() const {return this->_size_i;}
        uint16_t getWRows() const {return this->_size_o;}
        void loadWeights(FILE* fptr)
        {
            parseCSV(fptr, this->_W.get(), this->_size_i*this->_size_o);
        }
        void loadBias(FILE* fptr)
        {
            parseCSV(fptr, this->_B.get(), this->_size_o);
        }
        void loadWeights(const char* filename)
        {
            loadWeights(fopen(filename, "r"));
        }
        void loadBias(const char* filename)
        {
            loadBias(fopen(filename, "r"));
        }
        const char* id() const override {return this->_id;}
};

template<typename T = float>
class ReLuLayer final : public GenericLayer<T>
{
    private:
        friend class Net<T>;
        static const char _id[];
    public:
        ReLuLayer() = delete;
        ReLuLayer(const uint16_t &layer_len) : GenericLayer<T>(layer_len, layer_len){};
        ReLuLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) : GenericLayer<T>(layer_len, input_block, layer_len){};
        ReLuLayer(const GenericLayer<T> * prev_layer) : GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {};
        uint16_t getLayerLen() const {return this->_size_i;}
        void compute() override
        {
            for (size_t i = 0; i < this->_size_i; i++)
            {
                this->_out.get()[i] = this->_in.get()[i] > 0? this->_in.get()[i] : 0;
            }
        }
        const char* id() const override {return this->_id;}
};

template<typename T = float>
class NormLayer final : public GenericLayer<T>
{
    private:
        friend class Net<T>;
        static const char _id[];
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
        const char* id() const override {return this->_id;}
};

template<typename T = float>
class SoftMaxLayer final : public GenericLayer<T>
{
    private:
        friend class Net<T>;
        static const char _id[];
    public:
        SoftMaxLayer() = delete;
        SoftMaxLayer(const uint16_t &layer_len) : GenericLayer<T>(layer_len, layer_len){};
        SoftMaxLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) : GenericLayer<T>(layer_len, input_block, layer_len){};
        SoftMaxLayer(const GenericLayer<T>* prev_layer) : GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {};
        uint16_t getLayerLen() const {return this->_size_i;}
        
        void compute() override
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
        const char* id() const override {return this->_id;}
};

template<typename T = float>
class ConvLayer final : public GenericLayer<T>
{
    private:
        friend class Net<T>;
        static const char _id[];
        ConvKernel<T> _kernel;
        dim_t _dim;
        bool _extend = false;
    public:
        ConvLayer() = delete;
        ConvLayer(const dim_t &layer_dim) : _dim(layer_dim), GenericLayer<T>(layer_dim.cols*layer_dim.rows, layer_dim.cols*layer_dim.rows){};
        ConvLayer(const dim_t &layer_dim, const std::shared_ptr<T> &input_block) : _dim(layer_dim), GenericLayer<T>(layer_dim.cols*layer_dim.rows, input_block, layer_dim.cols*layer_dim.rows) {}
        ConvLayer(const GenericLayer<T>* prev_layer, const dim_t layer_dim) : _dim(layer_dim), GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {/*Error si dim =/= size */};

        uint16_t getLayerLen() const {return this->_size_i;}
        void setKernel(const ConvKernel<T> &kernel) {this->_kernel = kernel;}
        ConvKernel<T> getKernel() const {return _kernel;}
        void setPadding(ConvPadding padding)
        {
            _extend = padding == ConvPadding::SAME;
        }

        void compute() override
        {
            uint16_t i0 = this->_kernel.rows()/2;
            uint16_t j0 = this->_kernel.cols()/2;
            uint16_t iend = this->_dim.rows - i0;
            uint16_t jend = this->_dim.cols - j0;

            for (size_t i = 0; i < this->_dim.rows; i++)
            {
                for (size_t j = 0; j < this->_dim.cols; j++)
                {
                    if((i >= i0) && (i < iend) && (j >= j0) && (j < jend))
                    {
                        T calc = 0;
                        for (size_t ki = 0; ki < this->_kernel.rows(); ki++)
                        {
                            for (size_t kj = 0; kj < this->_kernel.cols(); kj++)
                            {
                                calc += this->_in.get()[this->_dim.rows*(i-i0+ki)+(j-j0+kj)] * this->_kernel.data[this->_kernel.rows()*ki+kj];
                            }
                        }
                        this->_out.get()[this->_dim.rows*i+j] = calc;
                    } 
                    else // Bordes
                    {
                        if (this->_extend) // SAME
                        {
                            if(i >= iend) // Borde inferior
                            {
                                if (i != j)
                                    this->_out.get()[this->_dim.rows*i+j] = this->_in.get()[this->_dim.rows*(iend-1)+j];
                                else
                                    this->_out.get()[this->_dim.rows*i+j] = this->_in.get()[this->_dim.rows*(iend-1)+(jend-1)];
                            }
                            else if (i < i0) // Borde superior
                            {
                                if (i != j)
                                    this->_out.get()[this->_dim.rows*i+j] = this->_in.get()[this->_dim.rows*i0+j];
                                else
                                    this->_out.get()[this->_dim.rows*i+j] = this->_in.get()[this->_dim.rows*i0+j0];
                            }
                            else if (j >= jend) // Borde derecho
                            {
                                this->_out.get()[this->_dim.rows*i+j] = this->_in.get()[this->_dim.rows*i+(jend-1)];
                            }
                            else // Borde izquierdo
                            {
                                this->_out.get()[this->_dim.rows*i+j] = this->_in.get()[this->_dim.rows*i+j0];
                            }
                        }
                        else // VALID
                        {
                            this->_out.get()[this->_dim.rows*i+j] = 0;
                        }
                    }
                }
            }
        }
        const char* id() const override {return this->_id;}
};


template<typename T = float>
class SigmoidLayer final : public GenericLayer<T>
{
    private:
        friend class Net<T>;
        static const char _id[];
    public:
        SigmoidLayer() = delete;
        SigmoidLayer(const uint16_t &layer_len) : GenericLayer<T>(layer_len, layer_len){};
        SigmoidLayer(const uint16_t &layer_len, const std::shared_ptr<T> &input_block) : GenericLayer<T>(layer_len, input_block, layer_len){};
        SigmoidLayer(const GenericLayer<T>* prev_layer) : GenericLayer<T>(prev_layer, prev_layer->getOutputSize()) {};
        uint16_t getLayerLen() const {return this->_size_i;}

        void compute() override
        {
            for (size_t i = 0; i < this->_size_i; i++)
            {
                this->_out.get()[i] = 1.0/(1.0+exp(-this->_in.get()[i]));
            }
        }
        const char* id() const override {return this->_id;}
};

template<typename T> const char GenericLayer<T>::_id[] = "Generic";
template<typename T> const char LambdaLayer<T>::_id[] = "Lambda";
template<typename T> const char WGLayer<T>::_id[] = "WG";
template<typename T> const char ReLuLayer<T>::_id[] = "ReLu";
template<typename T> const char NormLayer<T>::_id[] = "Normalize";
template<typename T> const char SoftMaxLayer<T>::_id[] = "SoftMax";
template<typename T> const char ConvLayer<T>::_id[] = "Convolution";
template<typename T> const char SigmoidLayer<T>::_id[] = "Sigmoid";

template<typename T>
class Net
{
    static_assert(std::is_floating_point<T>::value, "A Net class can only be instantiated with floating point types.");
    private:
        uint16_t _input_size, _output_size;
        std::vector<std::shared_ptr<GenericLayer<T>>> _layer_list;
        std::shared_ptr<T> _in;
        std::shared_ptr<T> _out;
    public:
        Net() = delete;
        Net(const uint16_t &input_len) : _input_size(input_len)
        {
            _in = std::shared_ptr<T>{new T[input_len]};
        }
        Net(const Net<T> &net) : _layer_list(std::move(net._layer_list))
        {
            this->_input_size = net._input_size;
            this->_output_size = net._output_size;
            this->_in = std::move(net._in);
            this->_out = std::move(net._out);
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
                _layer_list.emplace_back(new ReLuLayer<T>(_input_size, _in));
            }
            else
            {
                _layer_list.emplace_back(new ReLuLayer<T>(_layer_list.back().get()));
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

        // Convolve
        void addConvLayer(dim_t dimensions, ConvKernel<T> kernel, ConvPadding padding = ConvPadding::VALID)
        {            
            if (_layer_list.empty())
            {
                if(dimensions.cols*dimensions.rows != this->_input_size)
                {
                    // Error
                    dimensions = Dimensions(this->_input_size);
                }
                _layer_list.emplace_back(new ConvLayer<T>(dimensions, _in));
            }
            else
            {
                if(dimensions.cols*dimensions.rows != _layer_list.back()->_size_o)
                {
                    // Error
                    dimensions = Dimensions(_layer_list.back()->_size_o);
                }
                _layer_list.emplace_back(new ConvLayer<T>(_layer_list.back().get(),dimensions));
            }
            auto cptr = dynamic_cast<ConvLayer<T>*>(_layer_list.back().get());
            cptr->setKernel(kernel);
            cptr->setPadding(padding);
        }
        void addConvLayer(ConvKernel<T> kernel, ConvPadding padding = ConvPadding::VALID)
        {            
            dim_t dimensions{this->_input_size};
            addConvLayer(dimensions, kernel, padding);
        }

        // Sigmoide
        void addSigmoidLayer()
        {
            if (_layer_list.empty())
            {
                _layer_list.emplace_back(new SigmoidLayer<T>(_input_size, _in));
            }
            else
            {
                _layer_list.emplace_back(new SigmoidLayer<T>(_layer_list.back().get()));
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

        // Copiar entrada/salida
        void copy2input(const T* origin)
        {
            std::copy(origin, origin+_input_size, _in.get());
        }
        void copyout(T* dest)
        {
            std::copy(_out.get(), _out.get()+_output_size, dest);
        }

        // Block memory
        T* getInput() {return _in.get();}
        T* getInput() const {return _in.get();}
        T* getOutput() const {return _out.get();}

        // Lengths
        uint16_t getInputSize() const {return _input_size;};
        uint16_t getOutputSize() const {return _output_size;};

        auto tail() const {return _layer_list.back();}
        uint16_t n_layers() const {return _layer_list.size();}

};


template<typename T>
Net<T> loadNet(const char* toml_filename)
{
    auto docdata = toml::parse(toml_filename);
    auto& nn = toml::find(docdata, "NeuralNetwork");
    uint16_t inputs = toml::find<std::uint16_t>(nn, "inputs");
    uint16_t outputs = toml::find<std::uint16_t>(nn, "outputs");
    
    Net<T> net(inputs);

    auto& layers_data = toml::find(docdata, "Layers");
    uint16_t n_layers = toml::find<std::uint16_t>(layers_data, "size");
    
    uint16_t inlayer, outlayer, lenlayer; // Numer Layer Params 
    std::string r1, r2; // Route Layer Params 
    for (uint16_t i = 0; i < n_layers; i++)
    {
        char txt[5];
        sprintf(txt, "%d", (int)i);
        auto layer = toml::find(layers_data, txt);
        std::string type = toml::find<std::string>(layer, "type");
        
        if (type == "Normalize")
        {
            lenlayer = toml::find<std::uint16_t>(layer, "len");
            r1 = toml::find<std::string>(layer, "means");
            r2 = toml::find<std::string>(layer, "sd");
            if(net.n_layers() > 0)
            {
                if (net.tail()->getOutputSize() == lenlayer)
                {
                    // Ok
                    net.addNormLayer(r1.c_str(), r2.c_str());
                }
                else
                {
                    // Error
                    return net;
                }
            } 
            else
            {
                if (net.getInputSize() == lenlayer)
                {
                    // Ok
                    net.addNormLayer(r1.c_str(), r2.c_str());
                }
                else
                {
                    // Error
                    return net;
                }
            }
        } 
        else if  (type == "WG") 
        {
            inlayer = toml::find<std::uint16_t>(layer, "inputs");
            outlayer = toml::find<std::uint16_t>(layer, "outputs");
            r1 = toml::find<std::string>(layer, "weights");
            r2 = toml::find<std::string>(layer, "bias");
            if(net.n_layers() > 0)
            {
                if (net.tail()->getOutputSize() == inlayer)
                {
                    // Ok
                    net.addWGLayer(outlayer, r1.c_str(), r2.c_str());
                }
                else
                {
                    // Error
                    return net;
                }
            }
            else
            {
                if (net.getInputSize() == inlayer)
                {
                    // Ok
                    net.addWGLayer(outlayer, r1.c_str(), r2.c_str());
                }
                else
                {
                    // Error
                    return net;
                }
            }
        }
        else if (type == "ReLu")
        {
            lenlayer = toml::find<std::uint16_t>(layer, "len");
            if(net.n_layers() > 0)
            {
                if (net.tail()->getOutputSize() == lenlayer)
                {
                    // Ok
                    net.addReLuLayer();
                }
                else
                {
                    // Error
                    return net;
                }
            } 
            else
            {
                if (net.getInputSize() == lenlayer)
                {
                    // Ok
                    net.addReLuLayer();
                }
                else
                {
                    // Error
                    return net;
                }
            }
        }
        else if (type == "SoftMax")
        {
            lenlayer = toml::find<std::uint16_t>(layer, "len");
            if(net.n_layers() > 0)
            {
                if (net.tail()->getOutputSize() == lenlayer)
                {
                    // Ok
                    net.addSoftMaxLayer();
                }
                else
                {
                    // Error
                    return net;
                }
            } 
            else
            {
                if (net.getInputSize() == lenlayer)
                {
                    // Ok
                    net.addSoftMaxLayer();
                }
                else
                {
                    // Error
                    return net;
                }
            }
        }
        else if (type == "Sigmoid")
        {
            lenlayer = toml::find<std::uint16_t>(layer, "len");
            if(net.n_layers() > 0)
            {
                if (net.tail()->getOutputSize() == lenlayer)
                {
                    // Ok
                    net.addSigmoidLayer();
                }
                else
                {
                    // Error
                    return net;
                }
            } 
            else
            {
                if (net.getInputSize() == lenlayer)
                {
                    // Ok
                    net.addSigmoidLayer();
                }
                else
                {
                    // Error
                    return net;
                }
            }
        } 
        else 
        {
            // Error, capa no soportada
            return net;
        }
        

    }
    return net;
}




}

#endif