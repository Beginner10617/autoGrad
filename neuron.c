#include "neuron.h"
#include "autoGrad.h"
#include "stdio.h"
#include <stddef.h>
#include <stdlib.h>
// NEURON
Neuron *createNeuron(size_t sz, actFunc act) {
  Neuron *neuron = malloc(sizeof(Neuron));
  if (neuron == NULL) {
    printf("Unable to create neuron\n");
    exit(EXIT_FAILURE);
  }
  neuron->size = sz;
  neuron->activation = act;
  neuron->bias = doubleToValue((double)rand() / (double)RAND_MAX, true);
  neuron->weights = malloc(sizeof(Value *) * sz);
  if (neuron->weights == NULL) {
    printf("Unable to allocate space for weights\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < sz; i++) {
    neuron->weights[i] = doubleToValue((double)rand() / (double)RAND_MAX, true);
  }
  return neuron;
}
Value *setNeuron(Neuron *neuron, Value **inputs) {
  size_t size = neuron->size + 1;
  Value **intermediate = malloc(sizeof(Value *) * size);
  if (intermediate == NULL) {
    printf("Unable to allocate space for intermediate\n");
    exit(EXIT_FAILURE);
  }
  intermediate[size - 1] = neuron->bias;
  for (size_t i = 0; i < size - 1; i++) {
    intermediate[i] = EmptyValue(false);
    setMul(intermediate[i], neuron->weights[i], inputs[i]);
  }
  Value *output = EmptyValue(false);
  Value *tmp = EmptyValue(false);
  setSum(output, size);
  for (size_t i = 0; i < size; i++)
    addToSum(tmp, intermediate[i]);
  setTanh(output, tmp);
  return output;
}
// helper function
void prindActFunc(actFunc func) {
  if (func == none)
    printf("none");
  else if (func == tanh)
    printf("tanh");
}
void printNeuron(Neuron *neuron) {
  printf("activation : ");
  prindActFunc(neuron->activation);
  printf("\nBias : %f\n", neuron->bias->data);
  printf("weights : (%zu)\n", neuron->size);
  for (size_t i = 0; i < neuron->size; i++) {
    printf("%f, ", neuron->weights[i]->data);
  }
  printf("\n");
}

// LAYER
Layer *createLayer(size_t num_of_inputs, size_t num_of_outputs, actFunc act) {
  Layer *out = malloc(sizeof(Layer));
  if (out == NULL) {
    printf("Unable to allocate memory for layer\n");
    exit(EXIT_FAILURE);
  }
  out->num_of_neurons = num_of_outputs;
  out->size_of_neurons = num_of_inputs;
  out->neurons = malloc(sizeof(Neuron *) * num_of_outputs);
  if (out->neurons == NULL) {
    printf("UNable to allocate space for neurons in Layer\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < num_of_outputs; i++) {
    out->neurons[i] = createNeuron(num_of_inputs, act);
  }
  return out;
}

Value **setLayer(Layer *layer, Value **inputs) {
  Value **out = malloc(sizeof(Value *) * layer->num_of_neurons);
  for (size_t i = 0; i < layer->num_of_neurons; i++) {
    out[i] = setNeuron(layer->neurons[i], inputs);
  }
  return out;
}

void printLayer(Layer *layer) {
  printf("Number of neurons: %zu\nSize of each neuron: %zu\n",
         layer->num_of_neurons, layer->size_of_neurons);
  for (size_t i = 0; i < layer->num_of_neurons; i++)
    printNeuron(layer->neurons[i]);
}

// MLP
MLP *createMLP(size_t num_of_layers, size_t num_of_inputs,
               size_t *num_of_outputs, actFunc *acts) {
  MLP *out = malloc(sizeof(MLP));
  if (out == NULL) {
    printf("Unable to allocate space for MLP\n");
    exit(EXIT_FAILURE);
  }
  out->num_of_inputs = num_of_inputs;
  out->num_of_layers = num_of_layers;
  out->num_of_outputs = num_of_outputs;
  out->layers = malloc(sizeof(Layer *) * num_of_layers);
  if (out->layers == NULL) {
    printf("Unable to allocate space for layers in MLP\n");
    exit(EXIT_FAILURE);
  }
  out->layers[0] = createLayer(num_of_inputs, num_of_outputs[0], acts[0]);
  for (size_t i = 1; i < num_of_layers; i++) {
    out->layers[i] =
        createLayer(num_of_outputs[i - 1], num_of_outputs[i], acts[i]);
  }
  return out;
}

Value **setMLP(MLP *mlp, Value **inputs) {
  Value **curr_input = inputs;
  for (size_t i = 0; i < mlp->num_of_layers; i++) {
    curr_input = setLayer(mlp->layers[i], curr_input);
  }
  return curr_input;
}

void printMLP(MLP *mlp) {
  printf("Number of inputs: %zu\nNumber of layers: %zu\n", mlp->num_of_inputs,
         mlp->num_of_layers);
  printf("Outputs: ");
  for (size_t i = 0; i < mlp->num_of_layers; i++) {
    printf("%zu", mlp->num_of_outputs[i]);
    if (i < mlp->num_of_layers - 1)
      printf(", ");
    else
      printf("\n");
  }
  for (size_t i = 0; i < mlp->num_of_layers; i++)
    printLayer(mlp->layers[i]);
}
