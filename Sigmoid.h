#ifndef SIGMOID_H_INCLUDED
#define SIGMOID_H_INCLUDED

class Sigmoid : public Layer {
    public:
        Sigmoid(int nodes) {
            nxtSz = nodes;
        }

        double funct(double z) {
            return 1.0 / (1.0 + exp(-z));
        }

        double deriv(double z) {
            double s = funct(z);
            return s * (1.0 - s);
        }

        void feedForward(std::vector<double>& z, std::vector<double> a) override {
            for (int i = 0; i < z.size(); ++i) {
                z[i] = funct(a[i]);
            }
        }

        void backPropagation(std::vector<double>& error, std::vector<double> lastError, std::vector<double> z) override {
            for (int i = 0; i < error.size(); ++i) {
                error[i] = lastError[i] * deriv(z[i]);
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


#endif // SIGMOID_H_INCLUDED
