#ifndef DENSELAYER_H_INCLUDED
#define DENSELAYER_H_INCLUDED

class denseLayer : public Layer {
    public:
        int n, m;
        std::vector<std::vector<double>> weights, updNetWeights, velocityWeights;
        std::vector<double> biases, updNetBiases, velocityBiases;

        denseLayer(int _n, int _m) {
            n = _n, m = _m;
            sz = n;
            nxtSz = m;

            biases.resize(m);
            weights.resize(n);

            for (int i = 0; i < n; ++i) {
                weights[i].resize(m);
            }

            updNetBiases = biases;
            velocityBiases = biases;
            updNetWeights = weights;
            velocityWeights = weights;

            std::default_random_engine gen;
            std::normal_distribution<double> rngBiases(0.0, 1.0);

            for (int i = 0; i < m; ++i) {
                biases[i] = rngBiases(gen);
                updNetBiases[i] = biases[i];
                velocityBiases[i] = 0.0;
            }

            std::normal_distribution<double> rngWeights(0.0, 1.0 / sqrt(n));

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    weights[i][j] = rngWeights(gen);
                    updNetWeights[i][j] = weights[i][j];
                    velocityWeights[i][j] = 0.0;
                }
            }
        }

        void feedForward(std::vector<double>& z, std::vector<double> a) override {
            for (int i = 0; i < z.size(); ++i) {
                for (int j = 0; j < a.size(); ++j) {
                    z[i] += weights[j][i] * a[j];
                }

                z[i] += biases[i];
            }
        }

        void backPropagation(std::vector<double>& error, std::vector<double> lastError, std::vector<double> z) override {
            for (int i = 0; i < error.size(); ++i) {
                for (int j = 0; j < lastError.size(); ++j) {
                    error[i] += weights[i][j] * lastError[j];
                }
            }
        }

        void roll(std::vector<double> a, std::vector<double> error, double micro, double eta, int subsetSize) override {
            for (int i = 0; i < m; ++i) {
                (velocityBiases[i]) = (velocityBiases[i]) * micro - (eta / subsetSize) * error[i];
                (updNetBiases[i]) += (velocityBiases[i]);
                /// std::cout << i << " " << (velocityBiases[i]) << " " << micro << " " << (eta / subsetSize) << " " << error[i] << "\n";
            }

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    (velocityWeights[i][j]) = (velocityWeights[i][j]) * micro - (eta / subsetSize) * error[j] * a[i];
                    (updNetWeights[i][j]) += (velocityWeights[i][j]);
                }
            }
        }

        void regularize(double eta, double lambda, int trainDataSize) override {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    updNetWeights[i][j] -= ((eta * lambda) / trainDataSize) * weights[i][j];
                }
            }
        }

        void Update() override {
            for (int i = 0; i < m; ++i) {
                biases[i] = updNetBiases[i];
            }

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    weights[i][j] = updNetWeights[i][j];
                }
            }

            return;
        }
};

#endif // DENSELAYER_H_INCLUDED
