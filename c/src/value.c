#include <stdlib.h>
#include <math.h>
#include "value.h"

// Function prototypes for backward passes
static void backward_add(Value* self);
static void backward_mul(Value* self);
static void backward_relu(Value* self);

Value* Value_new(double data) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->backward = NULL;
    v->prev = NULL;
    v->n_prev = 0;
    v->prev_capacity = 0;
    v->op = (char*)"";  // Cast string literal to char* to satisfy C++
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

static void backward_add(Value* self) {
    self->prev[0]->grad += self->grad;
    self->prev[1]->grad += self->grad;
}

Value* Value_add(Value* a, Value* b) {
    Value* out = Value_new(a->data + b->data);
    Value_add_child(out, a);
    Value_add_child(out, b);
    out->op = (char*)"+";
    out->backward = backward_add;
    return out;
}

static void backward_mul(Value* self) {
    self->prev[0]->grad += self->prev[1]->data * self->grad;
    self->prev[1]->grad += self->prev[0]->data * self->grad;
}

Value* Value_mul(Value* a, Value* b) {
    Value* out = Value_new(a->data * b->data);
    Value_add_child(out, a);
    Value_add_child(out, b);
    out->op = (char*)"*";
    out->backward = backward_mul;
    return out;
}

static void backward_relu(Value* self) {
    self->prev[0]->grad += (self->data > 0) * self->grad;
}

Value* Value_relu(Value* a) {
    Value* out = Value_new(a->data < 0 ? 0 : a->data);
    Value_add_child(out, a);
    out->op = (char*)"ReLU";
    out->backward = backward_relu;
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
