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
#define XSTR2(x__) #x__
#define XSTR(x__) XSTR2(x__)

#include "neuronov_net.hpp"
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>
float random_number() {
    return ((float)(rand() * 2) / (float)RAND_MAX) - 1.0f;
}
float activation(float x_) {
    return x_ > 0 ? x_ : x_ * 0.1f;
}
float activation_d(float x_) {
    return x_ > 0 ? 1 : 0.1f;
}

bool learning_test() {
    std::cout << "learning test\n";
    neuronov_net::perseptron neuronet({1, 6, 6, 1}, activation, activation_d, random_number);
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
    std::cout << "real answer: " << realAnswer << "\nfeed forward result: ";
    for (auto i : neuronet.get_output()) {
        if (std::abs(i.value - realAnswer) > 0.15f) {
            std::cout << __FILE__ ":" XSTR(__LINE__) ": |out - real| > 0.15 - learning doesnt work as expect\n\n";
            return false;
        }
        std::cout << i.value << ',';
    }
    

    std::cout << "\ntrue\n\n";
    return true;
}
bool perfomance_test() {
    std::cout << "perfomance test\n";

    auto start = std::chrono::steady_clock::now();
    neuronov_net::perseptron neuronet({8, 30, 10, 3}, activation, activation_d, random_number);
    std::vector<float> correct(3);
    for (size_t i = 0; i < 100000; ++i) { // learn to copy sin
        float c = random_number() * 3.1415f;
        for (auto& n : neuronet.get_input())
            n.value = c;
        for (auto& n : correct)
            n = sin(c);

        neuronet.feed_forward();
        neuronet.learn(correct, 0.025f);
    }
    std::cout << "leaning time with arch {8, 30, 10, 3}: " << (double)(std::chrono::steady_clock::now() - start).count() / (double)std::chrono::steady_clock::period::den << '\n';

    start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 100000; ++i) {
        neuronet.feed_forward();
        
    }
    std::cout << "feed forward time with arch {8, 30, 10, 3} 100000 times: " << (double)(std::chrono::steady_clock::now() - start).count() / (double)std::chrono::steady_clock::period::den << '\n';
    
    std::cout << "true\n\n";
    return true;
}
bool safe_load_test() {
    std::cout << "safe/load test\n";

    std::stringstream saved1;
    neuronov_net::perseptron neuronet1({2, 5, 1}, activation, activation_d, random_number);
    neuronet1.safe(saved1);

    std::stringstream saved2;
    neuronov_net::perseptron neuronet2(activation, activation_d);
    neuronet2.load(saved1);
    neuronet2.safe(saved2);

    const auto str1 = saved1.str();
    const auto str2 = saved2.str();
    std::cout << str1 << '\n';
    std::cout << str2 << '\n';
    if (str1 != str2) {
        std::cout << __FILE__ ":" XSTR(__LINE__) ": str1 != str2 - load/safe failed\n\n";
        return false;
    }

    for (auto& n : neuronet1.get_input())
        n.value = 1;
    for (auto& n : neuronet2.get_input())
        n.value = 1;

    auto output1 = neuronet1.get_output();
    auto output2 = neuronet2.get_output();

    auto obeg1 = output1.begin();
    auto oend1 = output1.end();
    auto obeg2 = output2.begin();
    auto oend2 = output2.end();

    if (obeg1 == oend1) {
        std::cout << __FILE__ ":" XSTR(__LINE__) ": obeg1 == oend1 - something wrong\n\n";
        return false;
    }
    if (obeg2 == oend2) {
        std::cout << __FILE__ ":" XSTR(__LINE__) ": obeg2 == oend2 - something wrong\n\n";
        return false;
    }
    neuronet1.feed_forward();
    neuronet2.feed_forward();

    const float answer1 = output1[0].value;
    const float answer2 = output2[0].value;
    std::cout << "answer1 = " << answer1 << ", answer2 = " << answer2 << '\n'; 
    if (std::abs(answer1 - answer2) > 0.01f) {
        std::cout << __FILE__ ":" XSTR(__LINE__) ": answer1 != answer2 - load/safe failed\n\n";
        return false;
    }
    std::cout << "true\n\n";
    return true;
}
int main() {
    int success = 0;
    success += (int)learning_test();
    success += (int)safe_load_test();
    success += (int)perfomance_test();
    std::cout << success << "/3\n";
}
