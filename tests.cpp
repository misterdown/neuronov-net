/*  tests.cpp
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

#include "neuronov_net.hpp"
#include <iostream>
#include <random>
float random_number() {
    return ((float)(rand() * 2) / (float)RAND_MAX) - 1.0f;
}
float activation(float x_) {
    return x_ > 0 ? x_ : x_ * 0.1f;
}
float activation_d(float x_) {
    (void)x_;
    return x_ > 0 ? 1 : 0.1f;
}

bool global_test() {
    std::cout << "global test\n";

    neuronow_net::perseptron neuronet({1, 6, 6, 1}, activation, activation_d, random_number);
    std::vector<float> correct(1);
    for (size_t i = 0; i < 100000; ++i) { // learn to copy sin
        float c = random_number() * 3.1415f;
        for (auto& n : neuronet.get_input())
            n.value = c;
        for (auto& n : correct)
            n = sin(c);

        neuronet.feed_forward();
        neuronet.learn(correct, 0.025f);
    }
    const float realAnswer = sin(3.1415f / 4.0f);
    for (auto& i : neuronet.get_input())
        i.value = 3.1415f / 4.0f;

    neuronet.feed_forward();
    std::cout << "default feed forward: ";
    for (auto i : neuronet.get_output()) {
        if (std::abs(i.value - realAnswer) > 0.2)
            return false;
        std::cout << i.value << ',';
    }
    std::cout << '\n';

    auto compiled = neuronet.compile_to_programm();
    neuronet.execute(compiled);
    std::cout << "executing: ";
    for (auto i : neuronet.get_output()) {
        if (std::abs(i.value - realAnswer) > 0.15f)
            return false;
        std::cout << i.value << ',';
    }
    std::cout << '\n';
    std::cout << "real answer: " << realAnswer << '\n';
    return true;
}

int main() {
    int success = 0;
    success += (int)global_test();
    std::cout << success << "/1\n";

}