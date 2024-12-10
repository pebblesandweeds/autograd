#include <stdlib.h>
#include "../include/neuron.h"

static void neuron_zero_grad(void* self) {
    Neuron* n = (Neuron*)self;
    for (int i = 0; i < n->nin; i++) {
        n->w[i]->grad = 0;
    }
    n->b->grad = 0;
}

static Value** neuron_parameters(void* self, int* n_params) {
    Neuron* n = (Neuron*)self;
    *n_params = n->nin + 1;  // weights + bias
    Value** params = (Value**)malloc(*n_params * sizeof(Value*));
    for (int i = 0; i < n->nin; i++) {
        params[i] = n->w[i];
    }
    params[n->nin] = n->b;
    return params;
}

Neuron* Neuron_new(int nin, int nonlin) {
    Neuron* n = (Neuron*)malloc(sizeof(Neuron));
    n->nin = nin;
    n->nonlin = nonlin;
    
    // Initialize weights and bias
    n->w = (Value**)malloc(nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        double rand_val = ((double)rand() / RAND_MAX) * 2 - 1;
        n->w[i] = Value_new(rand_val);
    }
    n->b = Value_new(0);
    
    n->zero_grad = neuron_zero_grad;
    n->parameters = neuron_parameters;
    
    return n;
}

void Neuron_free(Neuron* n) {
    for (int i = 0; i < n->nin; i++) {
        Value_free(n->w[i]);
    }
    free(n->w);
    Value_free(n->b);
    free(n);
}

Value* Neuron_forward(Neuron* n, Value** x) {
    Value* act = n->b;  // Start with bias
    for (int i = 0; i < n->nin; i++) {
        Value* prod = Value_mul(n->w[i], x[i]);
        act = Value_add(act, prod);
    }
    return n->nonlin ? Value_relu(act) : act;
}
