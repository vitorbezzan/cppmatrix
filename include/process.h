#ifndef PROCESS_H
#define PROCESS_H

template<typename T>
class StochasticProcess {
public:
    StochasticProcess() = default;

    virtual ~StochasticProcess() = default;

    virtual T *path(size_t &timesteps) = 0;
};

#endif