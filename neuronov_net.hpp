/*  neuronov_net.hpp
MIT License

Copyright (c) 2024 Aidar Shigapov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// BRAIN ROTTING LIBRARY
#ifndef NEURONOV_NET_HPP_
#define NEURONOV_NET_HPP_

#ifndef NEURONOV_NET_DEFAULT_CONTAINER
#   include <vector>
#   define NEURONOV_NET_DEFAULT_CONTAINER ::std::vector
#endif

#ifndef NEURONOV_NET_SIZE_TYPE
#   include <cstdint>
#   define NEURONOV_NET_SIZE_TYPE ::std::size_t
#endif

#ifndef NEURONOV_NET_ASSERT
#   include <cassert>
#   define NEURONOV_NET_ASSERT(expr__) assert(expr__)
#endif

namespace neuronov_net {

// Forward declaration
template <template <class...> class ContainerT, class NumberT, class FunctionT>
struct perseptron_t;

// Safe view wrapper for layer access
template <class ContainerT>
struct LayerView {
    private:
    ContainerT* container;
    bool include_bias;
    
    public:
    typedef typename ContainerT::iterator iterator;
    typedef typename ContainerT::const_iterator const_iterator;
    typedef typename ContainerT::value_type value_type;
    typedef typename ContainerT::size_type size_type;
    typedef typename ContainerT::reference reference;
    typedef typename ContainerT::const_reference const_reference;

    LayerView() = delete;
    LayerView(const ContainerT& c, bool with_bias) 
        : container(const_cast<ContainerT*>(&c)), include_bias(with_bias) {}
        
    iterator begin() {
        return mut_container().begin(); 
    }
    
    const_iterator begin() const {
        return not_mut_container().begin();
    }
    
    iterator end() { 
        return include_bias ? mut_container().end() : mut_container().end() - 1; 
    }
    
    const_iterator end() const {
        return include_bias ? not_mut_container().end() : not_mut_container().end() - 1;
    }
    
    size_type size() const { 
        return include_bias ? not_mut_container().size() : not_mut_container().size() - 1; 
    }
    
    reference operator[](NEURONOV_NET_SIZE_TYPE idx) { 
        return mut_container()[idx]; 
    }
    
    const_reference operator[](NEURONOV_NET_SIZE_TYPE idx) const {
        return not_mut_container()[idx];
    }

    private:
    const ContainerT& not_mut_container() const {
        return *const_cast<const ContainerT*>(container);
    }
    ContainerT& mut_container() {
        return *const_cast< ContainerT*>(container);
    }
    
};
template <template <class...> class ContainerT = NEURONOV_NET_DEFAULT_CONTAINER, 
          class NumberT = float, 
          class FunctionT = NumberT(*)(NumberT)>
struct perseptron_t {
    private:
    struct neuron {
        NumberT value = 0;
        NumberT delta = 0;
    };
    
    using Layer = ContainerT<neuron>;
    using Weights = ContainerT<ContainerT<NumberT>>;
    using MLayerView = LayerView<Layer>;
    using CLayerView = const LayerView<const Layer>;
    
    ContainerT<Layer> layers;
    ContainerT<Weights> weights;
    FunctionT activation;
    FunctionT activationD;

	public:
    perseptron_t() = default;
    
    perseptron_t(FunctionT activation, FunctionT activationD) 
        : activation(activation), activationD(activationD) {}
    
    template<class CallableT_>
    perseptron_t(const ContainerT<NEURONOV_NET_SIZE_TYPE>& arch, FunctionT activation, FunctionT activationD, CallableT_ randomNumberGenerator) 
        : layers(arch.size()), 
          weights(arch.size() - 1),
          activation(activation),
          activationD(activationD) {
        NEURONOV_NET_ASSERT(arch.size() > 1);
        NEURONOV_NET_ASSERT(activation);
        NEURONOV_NET_ASSERT(activationD);

        for (NEURONOV_NET_SIZE_TYPE i = 0; i < arch.size(); ++i) {
            NEURONOV_NET_ASSERT(arch[i] > 0);

            const bool is_last = (i == (arch.size() - 1));
            auto& currentLayer = layers[i];
            currentLayer = ContainerT<neuron>(arch[i] + (is_last ? 0 : 1), neuron()); // bias on every layer, except last

            currentLayer.back().value = 1; // bias

            if (i >= 1) {
                weights[i - 1] = Weights(arch[i-1] + 1, ContainerT<NumberT>(arch[i])); // "+ 1" - bias
                for (auto& o : weights[i - 1])
                    for (auto& i : o) 
                        i = randomNumberGenerator();
            }
        }
    }

    void feed_forward() {
        for (NEURONOV_NET_SIZE_TYPE i = 0; i < layers.size() - 1; ++i) {
            const bool is_last = (i == layers.size() - 2);
            CLayerView current_view(layers[i], false);
            MLayerView next_view(layers[i + 1], is_last);
            
            for (NEURONOV_NET_SIZE_TYPE ni = 0; ni < next_view.size(); ++ni) {
                NumberT sum = 0;
                for (NEURONOV_NET_SIZE_TYPE ci = 0; ci < current_view.size(); ++ci) {
                    sum += current_view[ci].value * weights[i][ci][ni];
                }
                // Add bias contribution
                sum += weights[i].back()[ni]; // `layers[i].back().value * `
                next_view[ni].value = activation(sum);
            }
        }
    }
    
    void learn(const ContainerT<NumberT>& correctResults, NumberT learnRate) {
        NEURONOV_NET_ASSERT(correctResults.size() == layers.back().size());
    
        auto& output_layer = layers.back();
        for (NEURONOV_NET_SIZE_TYPE o = 0; o < output_layer.size(); ++o) {
            auto& neuron = output_layer[o];
            neuron.delta = (correctResults[o] - neuron.value) * activationD(neuron.value);
        }
        for (NEURONOV_NET_SIZE_TYPE rlayer_idx = 1; rlayer_idx < layers.size() - 1; ++rlayer_idx) {
            NEURONOV_NET_SIZE_TYPE layer_idx = layers.size() - rlayer_idx - 1;
            auto& current_layer = layers[layer_idx];
            auto& next_layer = layers[layer_idx + 1];
            auto& current_weights = weights[layer_idx];
            
            for (NEURONOV_NET_SIZE_TYPE i = 0; i < current_layer.size() - 1; ++i) {
                NumberT error = 0;
                for (NEURONOV_NET_SIZE_TYPE j = 0; j < next_layer.size(); ++j) {
                    error += current_weights[i][j] * next_layer[j].delta;
                }
                current_layer[i].delta = error * activationD(current_layer[i].value);
            }
            for (NEURONOV_NET_SIZE_TYPE i = 0; i < current_layer.size(); ++i) {
                for (NEURONOV_NET_SIZE_TYPE j = 0; j < next_layer.size(); ++j) {
                    current_weights[i][j] += learnRate * current_layer[i].value * next_layer[j].delta;
                }
            }
        }
    }
    template <class StreamT>
    void safe(StreamT& stream) const {
        for (const auto& i : layers) {
            stream << " " << i.size(); // with biases
        }
        stream << " 0 "; // Null terminator

        for (const auto& c : weights) {
            for (const auto& n : c) {
                for (const auto& v : n) {
                    stream << v << " ";
                }
            }
        }
    }
    template <class StreamT>
    void load(StreamT& stream) {
        // Read network architecture
        ContainerT<NEURONOV_NET_SIZE_TYPE> arch;

        while(true) {
            NEURONOV_NET_SIZE_TYPE lastSize;
            stream >> lastSize;
            if (lastSize == 0)
                break;
            arch.push_back(lastSize);
        }
        NEURONOV_NET_ASSERT(arch.size() > 1);
        
        layers = ContainerT<Layer>(arch.size());
        weights = ContainerT<Weights>(arch.size() - 1);

        for (NEURONOV_NET_SIZE_TYPE i = 0; i < arch.size(); ++i) {
            NEURONOV_NET_ASSERT(arch[i] > 0);

            const bool is_lasts = (i == (arch.size() - 1));
            auto& currentLayer = layers[i];
            currentLayer = Layer(arch[i]);

            currentLayer.back().value = 1; // bias

            if (i >= 1) {
                const NEURONOV_NET_SIZE_TYPE j = i - 1;
                weights[j] = Weights(arch[j], ContainerT<NumberT>(arch[i] - (is_lasts ? 0 : 1)));
                for (auto& wl : weights[j])
                    for (auto& w :wl)
                        stream >> w;
            }
        }
    }
    
    
    CLayerView get_input() const noexcept {
        return CLayerView(layers.front(), false);
    }
    MLayerView get_input() noexcept {
        return MLayerView(layers.front(), false);
    }
    CLayerView get_output() const noexcept {
        return CLayerView(layers.back(), true);
    }
    MLayerView get_output() noexcept {
        return MLayerView(layers.back(), true);
    }

};

typedef perseptron_t<> perseptron;

} // namespace neuronov_net
#endif // NEURONOV_NET_HPP_