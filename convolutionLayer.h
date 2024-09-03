#ifndef CONVOLUTIONLAYER_H_INCLUDED
#define CONVOLUTIONLAYER_H_INCLUDED

class convolutionLayer : public Layer {
    public:
        int n, m, k, maps;
        std::vector<std::vector<std::vector<double>>> weights, updNetWeights, velocityWeights;
        std::vector<double> biases, updNetBiases, velocityBiases;

        convolutionLayer(int _n, int _m, int _k, int _maps) {
            n = _n, m = _m, k = _k, maps = _maps;
            sz = n * m;
            nxtSz = maps * (n - k + 1) * (m - k + 1);

            biases.resize(maps);
            weights.resize(maps);

            for (int i = 0; i < maps; ++i) {
                weights[i].resize(k);

                for (int j = 0; j < k; ++j) {
                    weights[i][j].resize(k);
                }
            }

            updNetBiases = biases;
            velocityBiases = biases;
            updNetWeights = weights;
            velocityWeights = weights;

            std::default_random_engine gen;
            std::normal_distribution<double> rngBiases(0.0, 1.0);

            for (int i = 0; i < maps; ++ i) {
                biases[i] = rngBiases(gen);
                updNetBiases[i] = biases[i];
                velocityBiases[i] = 0.0;
            }

            std::normal_distribution<double> rngWeights(0.0, 1.0 / sqrt(k * k));

            for (int i = 0; i < maps; ++ i) {
                for (int j = 0; j < k; ++j) {
                    for (int l = 0; l < k; ++l) {
                        weights[i][j][l] = rngWeights(gen);
                        updNetWeights[i][j][l] = weights[i][j][l];
                        velocityWeights[i][j][l] = 0.0;
                    }
                }
            }
        }

        void feedForward(std::vector<double>& z, std::vector<double> a) override {
            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i < n - k + 1; ++i) {
                    for (int j = 0; j < m - k + 1; ++j) {
                        for (int p = i; p < i + k; ++p) {
                            for (int q = j; q < j + k; ++q) {
                                z[mp * (n - k + 1) * (m - k + 1) + i * (m - k + 1) + j] += weights[mp][p - i][q - j] * a[p * m + q];
                            }
                        }

                        z[mp * (n - k + 1) * (m - k + 1) + i * (m - k + 1) + j] += biases[mp];
                    }
                }
            }
        }

        void backPropagation(std::vector<double>& error, std::vector<double> lastError, std::vector<double> z) override {
            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i < n - k + 1; ++i) {
                    for (int j = 0; j < m - k + 1; ++j) {
                        for (int p = i; p < i + k; ++p) {
                            for (int q = j; q < j + k; ++q) {
                                error[p * m + q] += weights[mp][p - i][q - j] * lastError[mp * (n - k + 1) * (m - k + 1) + i * (m - k + 1) + j];
                            }
                        }
                    }
                }
            }
        }

        void roll(std::vector<double> a, std::vector<double> error, double micro, double eta, int subsetSize) override {
            for (int mp = 0; mp < maps; ++mp) {
                double sum = 0;

                for (int i = 0; i < n - k + 1; ++i) {
                    for (int j = 0; j < m - k + 1; ++j) {
                        sum += error[mp * (n - k + 1) * (m - k + 1) + i * (m - k + 1) + j];
                    }
                }

                (velocityBiases[mp]) = (velocityBiases[mp]) * micro - (eta / subsetSize) * sum;
                (updNetBiases[mp]) += (velocityBiases[mp]);
            }

            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i < k; ++i) {
                    for (int j = 0; j < k; ++j) {
                        double sum = 0;

                        for (int p = 0; p < n - k + 1; ++p) {
                            for (int q = 0; q < m - k + 1; ++q) {
                                int u = p + k, v = q + k;

                                sum += a[u * m + v] * error[mp * (n - k + 1) * (m - k + 1) + p * (m - k + 1) + q];
                            }
                        }

                        (velocityWeights[mp][i][j]) = (velocityWeights[mp][i][j]) * micro - (eta / subsetSize) * sum;
                        (updNetWeights[mp][i][j]) += (velocityWeights[mp][i][j]);
                    }
                }
            }
        }

        void regularize(double eta, double lambda, int trainDataSize) override {
            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i < k; ++i) {
                    for (int j = 0; j < k; ++j) {
                        updNetWeights[mp][i][j] -= ((eta * lambda) / trainDataSize) * weights[mp][i][j];
                    }
                }
            }
        }

        void Update() override {
            for (int i = 0; i < maps; ++i) {
                biases[i] = updNetBiases[i];
            }

            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i < k; ++i) {
                    for (int j = 0; j < k; ++j) {
                        weights[mp][i][j] = updNetWeights[mp][i][j];
                    }
                }
            }

            return;
        }
};

#endif // CONVOLUTIONLAYER_H_INCLUDED
