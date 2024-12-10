#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/mlp.h"

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Create a simple MLP: 2 -> 3 -> 1
    int nouts[] = {3, 1};
    MLP* mlp = MLP_new(2, nouts, 2);
    
    // Create input
    Value* x1 = Value_new(2.0);
    Value* x2 = Value_new(0.5);
    Value* x[] = {x1, x2};
    
    // Forward pass
    Value* out = MLP_forward(mlp, x);
    printf("Output: %f\n", out->data);
    
    // Backward pass
    Value_backward(out);
    
    // Get and print all parameters and their gradients
    int n_params;
    Value** params = mlp->parameters(mlp, &n_params);
    printf("Parameters and gradients:\n");
    for (int i = 0; i < n_params; i++) {
        printf("Param %d: value=%f, grad=%f\n", i, params[i]->data, params[i]->grad);
    }
    
    // Zero gradients
    mlp->zero_grad(mlp);
    
    // Cleanup
    free(params);
    Value_free(x1);
    Value_free(x2);
    Value_free(out);
    MLP_free(mlp);
    
    return 0;
}
