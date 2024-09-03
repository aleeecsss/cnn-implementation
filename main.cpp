#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <memory>

const double INF = 1e9;
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

#include "Layer.h"

#include "denseLayer.h"
#include "convolutionLayer.h"
#include "poolingLayer.h"

#include "ReLU.h"
#include "Sigmoid.h"
#include "softMax.h"

int layers;
std::vector<std::unique_ptr<Layer>> net;
std::vector<std::pair<std::vector<double>, std::vector<double>>> trainData, validationData, testData;

void learn(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& subset, double eta, double micro, double lambda) {
    for (const auto& test : subset) {
        std::vector<std::vector<double>> a(layers + 1), error(layers + 1);

        a[0].resize(test.first.size());
        error[0].resize(test.first.size());

        for (int j = 0; j < layers; ++j) {
            a[j + 1].resize(net[j] -> nxtSz, 0);
            error[j + 1].resize(net[j] -> nxtSz, 0);
        }

        std::copy(test.first.begin(), test.first.end(), a[0].begin());

        for (int j = 0; j < layers; ++j) {
            auto& layer = net[j];

            layer -> feedForward(a[j + 1], a[j]);
        }

        for (int k = 0; k < a[layers].size(); ++k) {
            error[layers][k] = a[layers][k] - test.second[k];
        }

        for (int j = layers; j > 0; --j) {
            auto& layer = net[j - 1];

            layer -> backPropagation(error[j - 1], error[j], a[j - 1]);
        }

        for (int j = 0; j < layers; ++j) {
            auto& layer = net[j];

            layer -> roll(a[j], error[j + 1], micro, eta, subset.size());
        }

        /// v <- v * micro - eta / subsetSize * error
        /// x <- x + v
    }

    for (int j = 0; j < layers; ++j) {
        auto& layer = net[j];

        layer -> regularize(eta, lambda, trainData.size());
    }

    /// x <- x - (eta * lambda / trainDataSize) * x

    for (int j = 0; j < layers; ++j) {
        auto& layer = net[j];

        layer -> Update();
    }
}

double Cost(std::vector<double> y, std::vector <double> a) {
    double ret = 0.0;

    for (int i = 0; i < y.size(); ++i) {
        ret += -(y[i] * log(a[i]));
    }

    return ret;
}

void calcAccuracy(std::vector<std::pair<std::vector<double>, std::vector<double>>>& t, int& correctTests, int& allTests, double& cost) {
    for (int i = 0; i < t.size(); ++i) {
        std::vector<std::vector<double>> a(layers + 1);

        a[0].resize(t[i].first.size());

        for (int j = 0; j < layers; ++j) {
            a[j + 1].resize(net[j] -> nxtSz, 0);
        }

        std::copy(t[i].first.begin(), t[i].first.end(), a[0].begin());

        for (int j = 0; j < layers; ++ j) {
            auto& layer = net[j];

            layer -> feedForward(a[j + 1], a[j]);
        }

        int predicted = 0, correctPos = 0;

        for (int j = 0; j < a[layers].size(); ++j) {
            if (a[layers][j] > a[layers][predicted]) {
                predicted = j;
            }
        }

        for (int j = 0; j < a[layers].size(); ++j) {
            if (t[i].second[j] > t[i].second[correctPos]) {
                correctPos = j;
            }
        }

        cost += Cost(t[i].second, a[layers]);

        if (predicted == correctPos) {
            ++correctTests;
        }

        ++allTests;
    }

    cost /= t.size();
}

void stochasticGradientDescent(int mode, int cntEpochs, int subsetSize, double eta, double micro, double lambda) {
    /// mode = 0: Choosing hyper-parameters.
    /// mode = 1: Training and testing the AI.

    if (mode == 0) {
        while (validationData.size() < testData.size()) {
            int lst = trainData.size() - 1;

            validationData.push_back(trainData[lst]);
            trainData.pop_back();
        }

        testData = validationData;
    }

    for (int ep = 0; ep < cntEpochs; ++ep) {
        std::shuffle(trainData.begin(), trainData.end(), rng);

        for (int i = 0; i < trainData.size(); i += subsetSize) {
            std::vector<std::pair<std::vector<double>, std::vector<double>>> subset;

            for (int j = i; j < std::min((int)trainData.size(), i + subsetSize); ++j) {
                subset.push_back(trainData[j]);
            }

            learn(subset, eta, micro, lambda);
        }

        int correctTests = 0, allTests = 0;
        double costTest = 0.0, costTrain = 0.0;

        calcAccuracy(testData, correctTests, allTests, costTest);

        std::cout << "Epoch " << ep << ":\n";
        std::cout << "Accuracy (Tests): ";
        std::cout << std::fixed << std::setprecision(3) << (100.0 * correctTests) / allTests << "%\n";

        correctTests = allTests = 0;

        calcAccuracy(trainData, correctTests, allTests, costTrain);

        std::cout << "Accuracy (Training): ";
        std::cout << std::fixed << std::setprecision(3) << (100.0 * correctTests) / allTests << "%\n";

        std::cout << "C = ";
        std::cout << std::fixed << std::setprecision(6) << costTest << " (Loss function for Tests, excluding L2 regularization)\n\n";
    }
}

std::ifstream fin;

void parseData(std::vector<std::pair<std::vector<double>, std::vector<double>>>& t, int lmt) {
    std::string S;

    while (fin >> S) {
        std::pair<std::vector<double>, std::vector<double>> test;
        std::vector<double> w;

        for (int ind = 1; ind < S.size(); ++ind) {
            if (S[ind] >= '0' && S[ind] <= '9') {
                int x = 0;
                while (S[ind] >= '0' && S[ind] <= '9') {
                    x = x * 10 + (S[ind] - '0');
                    ++ind;
                }
                w.push_back(x / 256.0);
            }
        }

        std::vector<double> y(10, 0.0);
        y[S[0] - '0'] = 1.0;

        test.first = w;
        test.second = y;

        t.push_back(test);

        if (t.size() > lmt - 1) {
            break;
        }
    }

    return;
}

int main() {
    /** net.push_back(std::make_unique<denseLayer>(784, 30));
    net.push_back(std::make_unique<Sigmoid>(30));
    net.push_back(std::make_unique<denseLayer>(30, 10));
    net.push_back(std::make_unique<softMax>(10)); **/

    net.push_back(std::make_unique<convolutionLayer>(28, 28, 5, 20));
    net.push_back(std::make_unique<ReLU>(11520));
    net.push_back(std::make_unique<poolingLayer>(24, 24, 2, 20));
    net.push_back(std::make_unique<denseLayer>(2880, 30));
    net.push_back(std::make_unique<ReLU>(30));
    net.push_back(std::make_unique<denseLayer>(30, 10));
    net.push_back(std::make_unique<softMax>(10));

    layers = net.size();

    fin.close();
    fin.open("data.txt");

    parseData(trainData, 60000);

    fin.close();
    fin.open("test.txt");

    parseData(testData, 10000);

    std::cout << "Reading done!\n";

    stochasticGradientDescent(1, 100, 10, 0.01, 0.5, 0.0);

    return 0;
}
