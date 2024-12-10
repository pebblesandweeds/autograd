# autograd

## Code Structure

```
micrograd_c/
├── include/
│   ├── value.h
│   ├── neuron.h
│   ├── layer.h
│   └── mlp.h
├── src/
│   ├── value.c
│   ├── neuron.c
│   ├── layer.c
│   ├── mlp.c
│   └── main.c
└── Makefile
```

## Design

```
MLP
 └─ Layers
     └─ Neurons
         └─ Values (weights & biases)
```
