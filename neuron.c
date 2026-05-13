#include "neuron.h"
#include "autoGrad.h"
#include "stdio.h"
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
  setSum(output, size);
  for (size_t i = 0; i < size; i++)
    addToSum(output, intermediate[i]);
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
