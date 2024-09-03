#ifndef SOFTMAX_H_INCLUDED
#define SOFTMAX_H_INCLUDED

class softMax : public Layer {
    public:
        softMax(int nodes) {
            nxtSz = nodes;
        }

        void funct(std::vector<double>& z, std::vector<double> a) {
            for (int j = 0; j < z.size(); ++j) { /// softMax
                double sumExp = 0.0;

                for (int k = 0; k < z.size(); ++k) {
                    sumExp += exp(a[k]);
                }

                z[j] = exp(a[j]) / sumExp;
            }

            return;
        }

        void feedForward(std::vector<double>& z, std::vector<double> a) override {
            funct(z, a);
        }

        void backPropagation(std::vector<double>& error, std::vector<double> lastError, std::vector<double> z) override {
            error = lastError;

            return;
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

#endif // SOFTMAX_H_INCLUDED
