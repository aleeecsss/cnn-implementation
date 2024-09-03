#ifndef POOLINGLAYER_H_INCLUDED
#define POOLINGLAYER_H_INCLUDED

class poolingLayer : public Layer {
    private:
        int n, m, p, q, l, maps;
        std::vector<int> argMax;
    public:
        poolingLayer(int _n, int _m, int _l, int _maps) {
            n = _n, m = _m, l = _l, maps = _maps;
            sz = n * m * maps;
            p = n / l, q = m / l;
            nxtSz = p * q * maps;
            argMax.resize(nxtSz);
        }

        void feedForward(std::vector<double>& z, std::vector<double> a) override {
            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i * l < n; ++i) {
                    for (int j = 0; j * l < m; ++j) {
                        double mx = -INF;
                        int ind = mp * n * m + i * l * m + j * l;

                        for (int u = i * l; u < (i + 1) * l; ++u) {
                            for (int v = j * l; v < (j + 1) * l; ++v) {
                                mx = std::max(mx, a[mp * n * m + u * m + v]);

                                if (a[ind] < a[mp * n * m + u * m + v]) {
                                    ind = mp * n * m + u * m + v;
                                }
                            }
                        }

                        argMax[mp * p * q + i * q + j] = ind;
                        z[mp * p * q + i * q + j] = mx;
                    }
                }
            }
        }

        void backPropagation(std::vector<double>& error, std::vector<double> lastError, std::vector<double> z) override {
            for (int mp = 0; mp < maps; ++mp) {
                for (int i = 0; i * l < n; ++i) {
                    for (int j = 0; j * l < m; ++j) {
                        error[argMax[mp * p * q + i * q + j]] = lastError[mp * p * q + i * q + j];
                    }
                }
            }
        }

        void roll(std::vector<double> a, std::vector<double> error, double micro, double eta, int subsetSize) override {
            return;
        }

        void regularize(double eta, double lambda, int trainDataSize) override {
            return;
        }

        void Update() override {
            return;
        }
};

#endif // POOLINGLAYER_H_INCLUDED
