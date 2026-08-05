#include "neuron.h"

// helper function
float randomFloat() { return ((double)rand() / RAND_MAX - 0.5f) * 0.05f; }

Neuron *createNeuron(int dim, const char *n) {
  Neuron *neuron = malloc(sizeof(Neuron));
  neuron->dimension = dim;
  strncpy(neuron->name, n, 4);
  neuron->name[5] = '\0';
  neuron->weights = malloc(sizeof(Value *) * dim);

  for (int i = 0; i < dim; i++) {
    neuron->weights[i] = createNewValue(randomFloat(), "w");
    snprintf(neuron->weights[i]->name, sizeof(neuron->weights[i]->name), "w%d",
             i);
    neuron->weights[i]->_isparameter =
        1; // Useful when freeing the computaion graph
    // printf("neuron weight %s = %f created\n", neuron->weights[i]->name,
    // neuron->weights[i]->data);
  }
  neuron->bias = createNewValue(randomFloat(), "b");
  neuron->bias->_isparameter = 1;
  return neuron;
}

Value *evaluateNeuron(Neuron *neuron, Value **inputs, int tanh) {

  Value *raw_act = neuron->bias;
  int dim = neuron->dimension;
  for (int i = 0; i < dim; i++) {
    if (inputs[i] == NULL) {
      // printf("Input dimension must match neuron\n");
      exit(1);
    }
    Value *wixi = mul(neuron->weights[i], inputs[i]);
    raw_act = add(raw_act, wixi);
  }
  if (tanh) // Implement other activation functions here (maybe in future)
    return vtanh(raw_act);
  return raw_act;
}

void freeComputationTree(Value *root) {
  NodeList *nodelist = createNodeList();
  buildTopoSort(nodelist, root);
  for (int i = nodelist->size - 1; i >= 0; i--) {
    if (nodelist->values[i]->_isparameter > 0)
      continue;
    // printf("Removing %s = Value(%f)\n", nodelist->values[i]->name,
    // nodelist->values[i]->data);
    free(nodelist->values[i]);
  }
  freeNodeList(nodelist);
}

void printn(Neuron *n) {
  printf("-------------------\n");
  printf("neuron name: %s\n", n->name);
  printf("dimension: %d\n", n->dimension);
  printf("bias: %f\n", n->bias->data);
  for (int i = 0; i < n->dimension; i++)
    printv(n->weights[i]);
  printf("-------------------\n");
}

void gradientDescentNeuron(Neuron *neuron, float stepSize) {
  for (int i = 0; i < neuron->dimension; i++)
    gradientDescentValue(neuron->weights[i], stepSize);
  gradientDescentValue(neuron->bias, stepSize);
}

void gradientDescentLayer(Layer *layer, float stepSize) {
  for (int i = 0; i < layer->num_of_neurons; i++)
    gradientDescentNeuron(layer->neurons[i], stepSize);
}

void gradientDescentMLP(MLP *mlp, float stepSize) {
  for (int i = 0; i < mlp->num_of_layers; i++)
    gradientDescentLayer(mlp->layers[i], stepSize);
}

Layer *createLayer(int num_of_inputs, int num_of_outputs, const char *n) {
  Layer *layer = malloc(sizeof(Layer));
  strncpy(layer->name, n, 4);
  layer->name[5] = '\0';
  layer->num_of_neurons = num_of_outputs;
  layer->dim_of_neurons = num_of_inputs;
  layer->neurons = malloc(sizeof(Neuron *) * num_of_outputs);

  for (int i = 0; i < num_of_outputs; i++) {
    layer->neurons[i] = createNeuron(num_of_inputs, "n");
    snprintf(layer->neurons[i]->name, sizeof(layer->neurons[i]->name), "n%d",
             i);
  }
  return layer;
}

Value **evaluateLayer(Layer *layer, Value **inputs, int tanh) {
  Value **output = malloc(sizeof(Value *) * layer->num_of_neurons);
  for (int i = 0; i < layer->num_of_neurons; i++) {
    output[i] = evaluateNeuron(layer->neurons[i], inputs, tanh);
  }
  return output;
}

void printl(Layer *layer) {
  printf("-------------------\n");
  printf("layer name: %s\n", layer->name);
  printf("%d neurons, \neach taking %d inputs\n", layer->num_of_neurons,
         layer->dim_of_neurons);
  for (int i = 0; i < layer->num_of_neurons; i++)
    printn(layer->neurons[i]);
  printf("-------------------\n");
}

MLP *createMLP(int num_of_ips, int num_of_layers, int *num_of_ops, int *actfunc,
               const char *n) {
  MLP *mlp = malloc(sizeof(MLP));
  mlp->actfunc = actfunc;
  strncpy(mlp->name, n, 4);
  mlp->name[5] = '\0';
  mlp->num_of_inputs = num_of_ips;
  mlp->num_of_layers = num_of_layers;
  mlp->layers = malloc(sizeof(Layer *) * num_of_layers);
  int ips = num_of_ips;
  for (int i = 0; i < num_of_layers; i++) {
    mlp->layers[i] = createLayer(ips, num_of_ops[i], "l");
    snprintf(mlp->layers[i]->name, sizeof(mlp->layers[i]->name), "l%d", i);
    ips = num_of_ops[i];
  }
  mlp->num_of_outputs = ips;
  return mlp;
}

Value **evaluateMLP(MLP *mlp, Value **inputs) {
  Value **ips = inputs;
  for (int i = 0; i < mlp->num_of_layers; i++) {
    ips = evaluateLayer(mlp->layers[i], ips, mlp->actfunc[i]);
  }
  return ips;
}

void printmlp(MLP *mlp) {
  printf("-------------------\n");
  printf("MLP name: %s\n", mlp->name);
  printf("%d layers\n", mlp->num_of_layers);
  printf("%d inputs\n%d outputs\n", mlp->num_of_inputs, mlp->num_of_outputs);
  for (int i = 0; i < mlp->num_of_layers; i++)
    printl(mlp->layers[i]);
  printf("-------------------\n");
}
