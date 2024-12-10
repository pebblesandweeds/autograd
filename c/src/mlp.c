#include <stdlib.h>
#include "mlp.h"

static void mlp_zero_grad(void* self) {
    MLP* mlp = (MLP*)self;
    for (int i = 0; i < mlp->n_layers; i++) {
        mlp->layers[i]->zero_grad(mlp->layers[i]);
    }
}

static Value** mlp_parameters(void* self, int* n_params) {
    MLP* mlp = (MLP*)self;
    *n_params = 0;

    // Count total parameters
    for (int i = 0; i < mlp->n_layers; i++) {
        int params_count;
        mlp->layers[i]->parameters(mlp->layers[i], &params_count);
        *n_params += params_count;
    }

    // Allocate and collect parameters
    Value** params = (Value**)malloc(*n_params * sizeof(Value*));
    int idx = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        int params_count;
        Value** layer_params_array = mlp->layers[i]->parameters(mlp->layers[i], &params_count);
        for (int j = 0; j < params_count; j++) {
            params[idx++] = layer_params_array[j];
        }
        free(layer_params_array);
    }
    return params;
}

MLP* MLP_new(int nin, int* nouts, int n_layers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->n_layers = n_layers;
    mlp->layers = (Layer**)malloc(n_layers * sizeof(Layer*));

    int current_nin = nin;
    for (int i = 0; i < n_layers; i++) {
        int nonlin = i != n_layers - 1;  // No ReLU on last layer
        mlp->layers[i] = Layer_new(current_nin, nouts[i], nonlin);
        current_nin = nouts[i];
    }

    mlp->zero_grad = mlp_zero_grad;
    mlp->parameters = mlp_parameters;

    return mlp;
}

void MLP_free(MLP* mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
        Layer_free(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

Value* MLP_forward(MLP* mlp, Value** x) {
    Value** current = x;
    for (int i = 0; i < mlp->n_layers; i++) {
        Value** next = Layer_forward(mlp->layers[i], current);
        if (i > 0) {
            free(current);
        }
        current = next;
    }
    // For simplicity, assume last layer has one output
    Value* result = current[0];
    free(current);
    return result;
}
