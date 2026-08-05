#ifndef NEURON_H
#define NEURON_H

#include "neuron_utils.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

typedef struct {
  Value **weights;
  Value *bias;
  int dimension;
  char name[6]; // for debugging
} Neuron;

// Creating a neuron with given dimension and name
Neuron *createNeuron(int dim, const char *name);

// Evaluating neuron activation value for given inputs
Value *evaluateNeuron(Neuron *neuron, Value **inputs, int tanh);

// Freeing allocated memory after backpropagation of non-parameter nodes
void freeComputationTree(Value *root);

// For debugging
void printn(Neuron *neuron);

void gradientDescentNeuron(Neuron *neuron, float stepSize);

typedef struct {
  char name[6]; // For debugging
  int num_of_neurons, dim_of_neurons;
  Neuron **neurons;
} Layer;

// Creating a layer with given lengths of input and output, and name
Layer *createLayer(int num_of_inputs, int num_of_outputs, const char *name);

// Evaluate layer for a given set of inputs
Value **evaluateLayer(Layer *layer, Value **inputs, int tanh);

// For debugging
void printl(Layer *layer);

void gradientDescentLayer(Layer *layer, float stepSize);

typedef struct {
  char name[6]; // For debugging
  Layer **layers;
  int num_of_outputs;
  int num_of_inputs, num_of_layers;
  int *actfunc;
} MLP;

// Creating a Multi-Layer-Perception, provided inputs and outputs of each layer
MLP *createMLP(int num_of_inputs, int num_of_layers, int *num_of_outs,
               int *actfunc, const char *name);

// Evaluate the entire MLP layer-by-layer on a given set of inputs
Value **evaluateMLP(MLP *mlp, Value **inputs);

// For debugging
void printmlp(MLP *mlp);

void gradientDescentMLP(MLP *mlp, float stepSize);

#endif
