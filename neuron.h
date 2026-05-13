// 1. Use set() to make the training computation tree
// 2. Topologically sort using the final node
// 3. Execute the training loop using the ValueList
#ifndef NEURON_H
#define NEURON_H
#include "autoGrad.h"
typedef struct Neuron Neuron;
typedef struct Layer Layer;
typedef struct MLP MLP;
typedef enum { none, tanh } actFunc;
struct Neuron {
  size_t size;
  Value **weights;
  Value *bias;
  actFunc activation;
};
Neuron *createNeuron(size_t sz, actFunc act);
Value *setNeuron(Neuron *neuron, Value **inputs);
void printNeuron(Neuron *neuron);

struct Layer {
  size_t num_of_neurons, size_of_neurons;
  Neuron **neurons;
};
Layer *createLayer(size_t num_of_inputs, size_t num_of_outputs);
Value **setLayer(Layer *layer, Value **inputs);
void printLayer(Layer *layer);

struct MLP {
  size_t *num_of_outputs;
  size_t num_of_inputs, num_of_layers;
  Layer *layers;
};
MLP *createMLP(size_t num_of_layers, size_t num_of_inputs,
               size_t *num_of_outputs);
Value *setMLP(MLP *mlp, Value **inputs);
void printMLP(MLP *mlp);
#endif
