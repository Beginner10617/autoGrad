#ifndef NEURON_H
#define NEURON_H
#include "autoGrad.h"
typedef struct Neuron Neuron;
typedef struct Layer Layer;
typedef struct MLP MLP;
enum actFunc { none, tanh };
struct Neuron {
  int size;
  Value **weights;
  Value *bias;
  actFunc activation;
};
Neuron *createNeuron(int sz, actFunc *act);
Value *evaluateNeuron(Neuron *neuron, Value **inputs);
void printNeuron(Neuron *neuron);

struct Layer {
  int num_of_neurons, size_of_neurons;
  Neuron **neurons;
};
Layer *createLayer(int num_of_inputs, int num_of_outputs);
Value **evaluateLayer(Layer *layer, Value **inputs);
void printLayer(Layer *layer);

struct MLP {
  int *num_of_outputs;
  int num_of_inputs, num_of_layers;
  Layer *layers;
};
MLP *createMLP(int num_of_layers, int num_of_inputs, int *num_of_outputs);
Value *evaluateMLP(MLP *mlp, Value **inputs);
void printMLP(MLP *mlp);
#endif
