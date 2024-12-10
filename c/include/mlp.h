#ifndef MLP_H
#define MLP_H

#include "layer.h"

typedef struct MLP {
    Layer** layers;
    int n_layers;
    ZeroGradFn zero_grad;
    ParametersFn parameters;
} MLP;

// Constructor/destructor
MLP* MLP_new(int nin, int* nouts, int n_layers);
void MLP_free(MLP* mlp);

// Operations
Value* MLP_forward(MLP* mlp, Value** x);

#endif // MLP_H
