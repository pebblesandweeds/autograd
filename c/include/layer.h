#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

typedef struct Layer {
    Neuron** neurons;
    int nin;        // Inputs to each neuron
    int nout;       // Number of neurons
    ZeroGradFn zero_grad;
    ParametersFn parameters;
} Layer;

// Constructor/destructor
Layer* Layer_new(int nin, int nout, int nonlin);
void Layer_free(Layer* l);

// Operations
Value** Layer_forward(Layer* l, Value** x);

#endif // LAYER_H
