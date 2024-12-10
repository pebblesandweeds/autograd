#include <stdlib.h>
#include "../include/layer.h"

static void layer_zero_grad(void* self) {
    Layer* l = (Layer*)self;
    for (int i = 0; i < l->nout; i++) {
        l->neurons[i]->zero_grad(l->neurons[i]);
    }
}

static Value** layer_parameters(void* self, int* n_params) {
    Layer* l = (Layer*)self;
    *n_params = 0;
    
    // First count total parameters
    for (int i = 0; i < l->nout; i++) {
        int neuron_params;
        l->neurons[i]->parameters(l->neurons[i], &neuron_params);
        *n_params += neuron_params;
    }
    
    // Allocate and collect parameters
    Value** params = (Value**)malloc(*n_params * sizeof(Value*));
    int idx = 0;
    for (int i = 0; i < l->nout; i++) {
        int neuron_params;
        Value** neuron_params = l->neurons[i]->parameters(l->neurons[i], &neuron_params);
        for (int j = 0; j < neuron_params; j++) {
            params[idx++] = neuron_params[j];
        }
        free(neuron_params);
    }
    return params;
}

Layer* Layer_new(int nin, int nout, int nonlin) {
    Layer* l = (Layer*)malloc(sizeof(Layer));
    l->nin = nin;
    l->nout = nout;
    
    // Create neurons
    l->neurons = (Neuron**)malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        l->neurons[i] = Neuron_new(nin, nonlin);
    }
    
    l->zero_grad = layer_zero_grad;
    l->parameters = layer_parameters;
    
    return l;
}

void Layer_free(Layer* l) {
    for (int i = 0; i < l->nout; i++) {
        Neuron_free(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

Value** Layer_forward(Layer* l, Value** x) {
    Value** out = (Value**)malloc(l->nout * sizeof(Value*));
    for (int i = 0; i < l->nout; i++) {
        out[i] = Neuron_forward(l->neurons[i], x);
    }
    return out;
}
