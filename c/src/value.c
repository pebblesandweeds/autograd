#include <stdlib.h>
#include <math.h>
#include "../include/value.h"

Value* Value_new(double data) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->backward = NULL;
    v->prev = NULL;
    v->n_prev = 0;
    v->prev_capacity = 0;
    v->op = "";
    return v;
}

void Value_free(Value* v) {
    if (v->prev) free(v->prev);
    free(v);
}

void Value_add_child(Value* v, Value* child) {
    if (v->n_prev >= v->prev_capacity) {
        v->prev_capacity = v->prev_capacity == 0 ? 2 : v->prev_capacity * 2;
        v->prev = (Value**)realloc(v->prev, v->prev_capacity * sizeof(Value*));
    }
    v->prev[v->n_prev++] = child;
}

Value* Value_add(Value* a, Value* b) {
    Value* out = Value_new(a->data + b->data);
    Value_add_child(out, a);
    Value_add_child(out, b);
    out->op = "+";
    
    void backward(Value* self) {
        a->grad += self->grad;
        b->grad += self->grad;
    }
    
    out->backward = backward;
    return out;
}

Value* Value_mul(Value* a, Value* b) {
    Value* out = Value_new(a->data * b->data);
    Value_add_child(out, a);
    Value_add_child(out, b);
    out->op = "*";
    
    void backward(Value* self) {
        a->grad += b->data * self->grad;
        b->grad += a->data * self->grad;
    }
    
    out->backward = backward;
    return out;
}

Value* Value_relu(Value* a) {
    Value* out = Value_new(a->data < 0 ? 0 : a->data);
    Value_add_child(out, a);
    out->op = "ReLU";
    
    void backward(Value* self) {
        a->grad += (out->data > 0) * self->grad;
    }
    
    out->backward = backward;
    return out;
}

static void build_topo(Value* v, Value** topo, int* topo_idx, Value** visited, int* visited_size) {
    for (int i = 0; i < *visited_size; i++) {
        if (visited[i] == v) return;
    }
    visited[(*visited_size)++] = v;
    for (int i = 0; i < v->n_prev; i++) {
        build_topo(v->prev[i], topo, topo_idx, visited, visited_size);
    }
    topo[(*topo_idx)++] = v;
}

void Value_backward(Value* v) {
    Value** topo = (Value**)malloc(1000 * sizeof(Value*));
    int topo_idx = 0;
    Value** visited = (Value**)malloc(1000 * sizeof(Value*));
    int visited_size = 0;
    
    build_topo(v, topo, &topo_idx, visited, &visited_size);
    v->grad = 1.0;
    
    for (int i = topo_idx - 1; i >= 0; i--) {
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }
    
    free(topo);
    free(visited);
}
