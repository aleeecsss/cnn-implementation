#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

class Layer {
    protected:
        int sz = 0, nxtSz = 0;
    public:
        virtual void feedForward(std::vector<double>&, std::vector<double>) = 0;
        virtual void backPropagation(std::vector<double>&, std::vector<double>, std::vector<double>) = 0;
        virtual void roll(std::vector<double>, std::vector<double>, double, double, int) = 0;
        virtual void regularize(double, double, int) = 0;
        virtual void Update() = 0;
        virtual ~Layer() = default;
        int get_nxtSz() {
            return nxtSz;
        }
};

#endif // LAYER_H_INCLUDED
