#ifndef VALUE_H
#define VALUE_H

typedef struct Value Value;
typedef void (*BackwardFn)(Value*);

struct Value {
    double data;
    double grad;
    BackwardFn backward;
    Value** prev;
    int n_prev;
    int prev_capacity;
    char* op;
};

// Constructor/destructor
Value* Value_new(double data);
void Value_free(Value* v);

// Operations
void Value_add_child(Value* v, Value* child);
Value* Value_add(Value* a, Value* b);
Value* Value_mul(Value* a, Value* b);
Value* Value_relu(Value* a);
void Value_backward(Value* v);

#endif // VALUE_H
