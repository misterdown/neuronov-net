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
#   define NEURONOV_NET_HPP_ 1
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
    typedef char byte;
    static_assert(sizeof(byte) == 1);

    template<class ContainerT_>
    struct container_view {
        public:
        typedef typename ContainerT_::iterator iterator;
        typedef typename ContainerT_::value_type value_type;
        typedef typename ContainerT_::reference reference;
        typedef typename ContainerT_::const_reference const_reference;

        private:
        iterator begin_;
        iterator end_;

        public:
        container_view() : begin_(), end_() {

        }
        container_view(iterator begin__, iterator end__) : begin_(begin__), end_(end__) {

        }

        public:
        /**
         * @brief Returns the beginning iterator of the container view.
         * @return Iterator to the beginning of the container view.
         */
        [[nodiscard]] iterator begin() noexcept {
            return begin_;
        }
        /**
         * @brief Returns the ending iterator of the container view.
         * @return Iterator to the end of the container view.
         */
        [[nodiscard]] iterator end() noexcept {
            return end_;
        }
        [[nodiscard]] reference operator[](NEURONOV_NET_SIZE_TYPE index) {
            return *(begin_ + index);
        }
        [[nodiscard]] const_reference operator[](NEURONOV_NET_SIZE_TYPE index) const {
            return *(begin_ + index);
        }

    };
    template<class ContainerT_>
    struct const_container_view {
        public:
        typedef typename ContainerT_::const_iterator const_iterator;
        typedef typename ContainerT_::value_type value_type;
        typedef typename ContainerT_::const_reference const_reference;

        private:
        const_iterator begin_;
        const_iterator end_;

        public:
        const_container_view() : begin_(), end_() {

        }
        const_container_view(const_iterator begin__, const_iterator end__) : begin_(begin__), end_(end__) {

        }

        public:
        /**
         * @brief Returns the beginning const iterator of the container view.
         * @return Iterator to the beginning of the container view.
         */
        [[nodiscard]] const_iterator begin() const noexcept {
            return begin_;
        }
        /**
         * @brief Returns the ending const iterator of the container view.
         * @return Iterator to the end of the container view.
         */
        [[nodiscard]] const_iterator end() const noexcept {
            return end_;
        }
        [[nodiscard]] const_reference operator[](NEURONOV_NET_SIZE_TYPE index) const {
            return *(begin_ + index);
        }

    };
    template <template <class...> class ContainerT_ = NEURONOV_NET_DEFAULT_CONTAINER, class NumberT_ = float, class FunctionT_ = NumberT_(*)(NumberT_)>
    struct perseptron_t {
        private:
        struct neuron {
            public:
            NumberT_ value;
            NumberT_ delta;

            public:
            neuron() : value(0), delta(0) {

            }

        };

        private:
        typedef container_view<ContainerT_<neuron>> icontainer_view_type;
        typedef const_container_view<ContainerT_<neuron>> iconst_container_view_type;

        private:
        ContainerT_<ContainerT_<neuron>> layers_;
        ContainerT_<ContainerT_<ContainerT_<NumberT_>>> weigths_; // layer | output neuron | input neuron
        FunctionT_ activation_;
        FunctionT_ activationD_;

        public:
        perseptron_t() {

        }
        perseptron_t(FunctionT_ activation, FunctionT_ activationD) : activation_(activation), activationD_(activationD) {

        }
        template<class CallableT_>
        perseptron_t(const ContainerT_<NEURONOV_NET_SIZE_TYPE>& arch, FunctionT_ activation, FunctionT_ activationD, CallableT_ randomNumberGenerator) : layers_(arch.size()), weigths_(arch.size() - 1), activation_(activation), activationD_(activationD) {
            NEURONOV_NET_ASSERT(arch.size() > 1);
            NEURONOV_NET_ASSERT(activation_);
            NEURONOV_NET_ASSERT(activationD_);

            for (NEURONOV_NET_SIZE_TYPE i = 0; i < arch.size(); ++i) {
                NEURONOV_NET_ASSERT(arch[i] > 0);

                const bool isOutputLayer = (i == (arch.size() - 1));
                auto& currentLayer = layers_[i];
                currentLayer = ContainerT_<neuron>(arch[i] + (isOutputLayer ? 0 : 1), neuron()); // bias on every layer, except last

                if (!isOutputLayer)
                    currentLayer.back().value = 1;// bias

                if (i >= 1) {
                    weigths_[i - 1] = ContainerT_<ContainerT_<NumberT_>>(arch[i-1] + 1, ContainerT_<NumberT_>(arch[i])); // "+ 1" - bias
                    for (auto& o : weigths_[i - 1])
                        for (auto& i : o) 
                            i = randomNumberGenerator();
                }
            }
        }

        public:
        /**
         * @brief Performs a feed-forward pass through the neural network.
         */
        void feed_forward() {
            for (NEURONOV_NET_SIZE_TYPE i = 0; i < layers_.size() - 1; ++i) {
                const auto& currentNeurons = layers_[i];
                auto& nextNeurons = layers_[i + 1];
                const NEURONOV_NET_SIZE_TYPE nextSize = (i == (layers_.size() - 2)) ? nextNeurons.size() : (nextNeurons.size() - 1); // skip bias

                for (NEURONOV_NET_SIZE_TYPE ni = 0; ni < nextSize; ++ni) { 
                    auto& nv = nextNeurons[ni].value;
                    nv = 0;
                    for (NEURONOV_NET_SIZE_TYPE ci = 0; ci < currentNeurons.size(); ++ci)
                        nv += currentNeurons[ci].value * weigths_[i][ci][ni];
                    nv = activation_(nv);
                }
            }
        }
        /**
         * @brief Performs a learning step for the neural network.
         * @param correctResults the correct results for the output layer.
         * @param learnRate the learning rate for adjusting weights.
         */
        void learn(const ContainerT_<NumberT_>& correctResults, NumberT_ learnRate) {
            auto& output = layers_.back();
            NEURONOV_NET_ASSERT(correctResults.size() == output.size());
            for (NEURONOV_NET_SIZE_TYPE o = 0; o < output.size(); ++o) {
                auto& n = output[o];
                n.delta = correctResults[o] - n.value;
            }

            for (NEURONOV_NET_SIZE_TYPE ii = layers_.size() -  1; ii > 0; --ii) {
                const NEURONOV_NET_SIZE_TYPE i = ii - 1;
                auto& currentNeurons = layers_[i];
                const auto& nextNeurons = layers_[i + 1];
                for (NEURONOV_NET_SIZE_TYPE ci = 0; ci < currentNeurons.size(); ++ci) { 
                    auto& n = currentNeurons[ci];
                    const NEURONOV_NET_SIZE_TYPE nextSize = (i == (layers_.size() - 2)) ? nextNeurons.size() : (nextNeurons.size() - 1); // skip bias

                    n.delta = 0;
                    auto& currentWeigths = weigths_[i][ci];
                    for (NEURONOV_NET_SIZE_TYPE ni = 0; ni < nextSize; ++ni)
                        n.delta += nextNeurons[ni].delta * currentWeigths[ni];
                    n.delta *= activationD_(n.value);
                    for (NEURONOV_NET_SIZE_TYPE ni = 0; ni < nextSize; ++ni)                //
                        currentWeigths[ni] += n.value * nextNeurons[ni].delta * learnRate;  //
                }
            }
        }
        /**
         * @brief Saves the neural network architecture and weights to a stream.
         * @param stream the output stream to save the data to.
         */
        template<class StreamT_>
        void safe(StreamT_& stream) const {
            NEURONOV_NET_ASSERT(layers_.size() > 1);
            NEURONOV_NET_ASSERT(weigths_.size() > 0);

            for (NEURONOV_NET_SIZE_TYPE i = 0; i < layers_.size() - 1; ++i)
                stream << layers_[i].size() << ' ';
            stream << layers_.back().size() << " 0 ";
            for (NEURONOV_NET_SIZE_TYPE i = 0; i < layers_.size() - 1; ++i) {
                for (const auto& wl : weigths_[i]) {
                    for (const auto& w :wl) {
                        stream << w << ' ';
                    }
                }
            }
        }
         /**
         * @brief Loads the neural network architecture and weights from a stream.
         * @param stream the input stream to load the data from.
         */
        template<class StreamT_>
        void load(StreamT_& stream) {
            NEURONOV_NET_ASSERT(activation_);
            NEURONOV_NET_ASSERT(activationD_);

            ContainerT_<NEURONOV_NET_SIZE_TYPE> arch;

            while(true) {
                NEURONOV_NET_SIZE_TYPE lastSize;
                stream >> lastSize;
                if (lastSize == 0)
                    break;
                arch.push_back(lastSize);
            }
            NEURONOV_NET_ASSERT(arch.size() > 1);
            
            layers_ = ContainerT_<ContainerT_<neuron>>(arch.size());
            weigths_ = ContainerT_<ContainerT_<ContainerT_<NumberT_>>>(arch.size() - 1);


            for (NEURONOV_NET_SIZE_TYPE i = 0; i < arch.size(); ++i) {
                NEURONOV_NET_ASSERT(arch[i] > 0);

                const bool isOutputLayer = (i == (arch.size() - 1));
                auto& currentLayer = layers_[i];
                currentLayer = ContainerT_<neuron>(arch[i], neuron());

                if (!isOutputLayer)
                    currentLayer.back().value = 1;// bias

                if (i >= 1) {
                    const NEURONOV_NET_SIZE_TYPE j = i - 1;
                    weigths_[j] = ContainerT_<ContainerT_<NumberT_>>(arch[j], ContainerT_<NumberT_>(arch[i] - (isOutputLayer ? 0 : 1)));
                    for (auto& wl : weigths_[j])
                        for (auto& w :wl)
                            stream >> w;
                }
            }
        }

        public:
        /**
         * @brief Returns a constant view of the output layer.
         * @return A constant container view of the output layer.
         */
        [[nodiscard]] iconst_container_view_type get_output() const noexcept {
            return iconst_container_view_type(layers_.back().begin(), layers_.back().end());
        }
        /**
         * @brief Returns a view of the input layer.
         * @return A container view of the input layer.
         */
        [[nodiscard]] icontainer_view_type get_input() noexcept {
            return icontainer_view_type(layers_.front().begin(), layers_.front().end() - 1);
        }
        /**
         * @brief Returns a constant view of the input layer.
         * @return A constant container view of the input layer.
         */
        [[nodiscard]] iconst_container_view_type get_const_input() const noexcept {
            return iconst_container_view_type(layers_.front().begin(), layers_.front().end() - 1);
        }

    };

    // Почему я должен оставлять "<>". Почему.

    typedef perseptron_t<> perseptron;
    
};
#endif // ifndef NEURONOV_NET_HPP_
