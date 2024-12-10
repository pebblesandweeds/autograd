#ifndef NEURON_H
#define NEURON_H

#include "value.h"

typedef struct Neuron Neuron;
typedef void (*ZeroGradFn)(void*);
typedef Value** (*ParametersFn)(void*, int*);

struct Neuron {
    Value** w;      // Weights
    Value* b;       // Bias
    int nin;        // Number of inputs
    int nonlin;     // Whether to use ReLU
    ZeroGradFn zero_grad;
    ParametersFn parameters;
};

// Constructor/destructor
Neuron* Neuron_new(int nin, int nonlin);
void Neuron_free(Neuron* n);

// Operations
Value* Neuron_forward(Neuron* n, Value** x);

#endif // NEURON_H
