#ifndef AUTOGRAD
#define AUTOGRAD
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#define ERROR "\x1b[31m"
#define RESET "\x1b[0m"
typedef struct Value Value;
typedef void (*Funcptr)(Value *);
struct Value {
  double data, grad;
  Funcptr _backward, _forward;
  struct Value **_prev;
  size_t _prevsz, _prevcap;
  bool _modifiable;
};
// Constructors
Value *EmptyValue(bool modify);
Value *floatToValue(float x, bool modify);
Value *doubleToValue(double x, bool modify);
// set out = x <op> y
void setAdd(Value *out, Value *x, Value *y);
void setSub(Value *out, Value *x, Value *y);
void setMul(Value *out, Value *x, Value *y);
void setTanh(Value *out, Value *in);
// to be used together:
void setSum(Value *out, size_t size);
void addToSum(Value *out, Value *x);
// _forward
void _addFwd(Value *x);
void _subFwd(Value *x);
void _mulFwd(Value *x);
void _sumFwd(Value *x);
void _tanhFwd(Value *x);
// _backward
void _addBack(Value *x);
void _subBack(Value *x);
void _mulBack(Value *x);
void _sumBack(Value *x);
void _tanhBack(Value *x);
// null function
void doNothing(Value *x);
// print
void printValue(Value *x);
// Destructors
void Destroy(Value **x);

// 1. Use set() to make the training computation tree
// 2. Topologically sort using the final node
// 3. Execute the training loop using the ValueList
typedef struct Neuron Neuron;
typedef struct Layer Layer;
typedef struct MLP MLP;
typedef enum { none, _tanh } actFunc;
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
  actFunc activation;
};
Layer *createLayer(size_t num_of_inputs, size_t num_of_outputs, actFunc act);
Value **setLayer(Layer *layer, Value **inputs);
void printLayer(Layer *layer);

struct MLP {
  size_t *num_of_outputs;
  size_t num_of_inputs, num_of_layers;
  Layer **layers;
};
MLP *createMLP(size_t num_of_layers, size_t num_of_inputs,
               size_t *num_of_outputs, actFunc *acts);
Value **setMLP(MLP *mlp, Value **inputs);
void printMLP(MLP *mlp);

typedef struct ValueList ValueList;

// a list to hold topo-sorted list of values, executing fwd-bwd passes on it
struct ValueList {
  Value **values;
  size_t size, _cap;
};
ValueList *CreateValueList();
// helper function
void appendValue(ValueList *lst, Value *val);
Value *getValueAt(ValueList *lst, size_t index);
// toposort
void topoSortList(Value *val, ValueList *lst);
// fwd-bwd passes, compute topoSortList once and use multiple times
void forward(ValueList *lst);
void backward(ValueList *lst);
// modifying values - gradient descent
void gradientDescent(ValueList *lst, double learningRate);

#ifdef AUTOGRAD_IMPLEMENTATION
// Constructors
Value *EmptyValue(bool modify) {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
#ifdef DEBUG
    printf(ERROR "E01: Error allocating space for value\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->data = 0;
  out->grad = 0;
  out->_backward = doNothing;
  out->_forward = doNothing;
  out->_prev = NULL;
  out->_prevsz = 0;
  out->_prevcap = 0;
  out->_modifiable = modify;
  return out;
}

Value *floatToValue(float x, bool modify) {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
#ifdef DEBUG
    printf(ERROR "E02: Error allocating space for value\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->data = x;
  out->grad = 0;
  out->_backward = doNothing;
  out->_forward = doNothing;
  out->_prev = NULL;
  out->_prevsz = 0;
  out->_prevcap = 0;
  out->_modifiable = modify;
  return out;
}

Value *doubleToValue(double x, bool modify) {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
#ifdef DEBUG
    printf(ERROR "E03: Error allocating space for value\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->data = x;
  out->grad = 0;
  out->_backward = doNothing;
  out->_forward = doNothing;
  out->_prev = NULL;
  out->_prevsz = 0;
  out->_prevcap = 0;
  out->_modifiable = modify;
  return out;
}

// set out = x <op> y
void setAdd(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
#ifdef DEBUG
    printf(ERROR "E04: NULL passed to setAdd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_forward = _addFwd;
  out->_backward = _addBack;
  out->_prevsz = 2;
  out->_prevcap = 2;
  out->_prev = malloc(sizeof(Value *) * 2);
  if (out->_prev == NULL) {
#ifdef DEBUG
    printf(ERROR "E05: Unable to allocate _prev inside setAdd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = x;
  out->_prev[1] = y;
}

void setSub(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
#ifdef DEBUG
    printf(ERROR "E06: NULL passed to setSub\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_forward = _subFwd;
  out->_backward = _subBack;
  out->_prevsz = 2;
  out->_prevcap = 2;
  out->_prev = malloc(sizeof(Value *) * 2);
  if (out->_prev == NULL) {
#ifdef DEBUG
    printf(ERROR "E07: Unable to allocate _prev inside setSub\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = x;
  out->_prev[1] = y;
}

void setMul(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
#ifdef DEBUG
    printf(ERROR "E08: NULL passed to setMul\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_forward = _mulFwd;
  out->_backward = _mulBack;
  out->_prevsz = 2;
  out->_prevcap = 2;
  out->_prev = malloc(sizeof(Value *) * 2);
  if (out->_prev == NULL) {
#ifdef DEBUG
    printf(ERROR "E09: Unable to allocate _prev inside setMul\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = x;
  out->_prev[1] = y;
}
void setTanh(Value *out, Value *in) {
  if (out == NULL || in == NULL) {
#ifdef DEBUG
    printf(ERROR "E10: NULL passed to setTanh\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_backward = _tanhBack;
  out->_forward = _tanhFwd;
  out->_prevsz = 1;
  out->_prevcap = 1;
  out->_prev = malloc(sizeof(Value *));
  if (out->_prev == NULL) {
#ifdef DEBUG
    printf(ERROR "E11: Unable to allocate _prev inside setTanh\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = in;
}
void setSum(Value *out, size_t size) {
  if (out == NULL || size == 0) {
#ifdef DEBUG
    printf(ERROR "E12: NULL passed to setSum\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_forward = _sumFwd;
  out->_backward = _sumBack;
  out->_prevsz = 0;
  out->_prevcap = size;
  out->_prev = malloc(sizeof(Value *) * out->_prevcap);
  if (out->_prev == NULL) {
#ifdef DEBUG
    printf(ERROR "E13: Unable to allocate _prev inside setSum\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
}
void addToSum(Value *out, Value *x) {
  if (out->_prevsz >= out->_prevcap) {
#ifdef DEBUG
    printf(ERROR "E14: Parameters to sum overflowed\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E15: NULL passed to addToSum\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->_prev[out->_prevsz] = x;
  out->_prevsz++;
}

// _forward : evaluate out = x <op> y
void _addFwd(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E16: NULL passed to _addFwd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
#ifdef DEBUG
    printf(ERROR "E17: Argument of add not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data + x->_prev[1]->data;
}

void _subFwd(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E18: NULL passed to _subFwd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
#ifdef DEBUG
    printf(ERROR "E19: Argument of sub not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data - x->_prev[1]->data;
}

void _mulFwd(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E20: NULL passed to _mulFwd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
#ifdef DEBUG
    printf(ERROR "E21: Argument of mul not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data * x->_prev[1]->data;
}
void _sumFwd(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E22: NULL passed to _sumFwd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->data = 0;
  for (size_t i = 0; i < x->_prevcap; i++) {
    if (x->_prev[i] == NULL) {
#ifdef DEBUG
      printf(ERROR "E23: Argument of sum not set, NULL encountered\n" RESET);
#endif
      exit(EXIT_FAILURE);
    }
    x->data += x->_prev[i]->data;
  }
}
void _tanhFwd(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E24: NULL passed to _tanhFwd\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x->_prev == NULL) {
#ifdef DEBUG
    printf(ERROR "E25: Argument of tanh not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->data = tanh(x->_prev[0]->data);
}

// _backward
void _addBack(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E26: NULL passed to _addBack\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
#ifdef DEBUG
    printf(ERROR "E27: Argument of add not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->_prev[0]->grad += x->grad;
  x->_prev[1]->grad += x->grad;
}

void _subBack(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E28: NULL passed to _subBack\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
#ifdef DEBUG
    printf(ERROR "E29: Argument of sub not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->_prev[0]->grad += x->grad;
  x->_prev[1]->grad -= x->grad;
}

void _mulBack(Value *z) {
  if (z == NULL) {
#ifdef DEBUG
    printf(ERROR "E30: NULL passed to _mulBack\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  Value *x = z->_prev[0];
  Value *y = z->_prev[1];
  if (x == NULL || y == NULL) {
#ifdef DEBUG
    printf(ERROR "E31: Argument of mul not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  x->grad += y->data * z->grad;
  y->grad += x->data * z->grad;
}
void _sumBack(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E32: NULL passed to _sumBack\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < x->_prevcap; i++) {
    if (x->_prev[i] == NULL) {
#ifdef DEBUG
      printf(ERROR "E33: Argument of sum not set, NULL encountered\n" RESET);
#endif
      exit(EXIT_FAILURE);
    }
    x->_prev[i]->grad += x->grad;
  }
}
void _tanhBack(Value *x) {
  if (x == NULL) {
#ifdef DEBUG
    printf(ERROR "E34: NULL passed to _tanhBack\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  Value *y = x->_prev[0];
  if (x == NULL || y == NULL) {
#ifdef DEBUG
    printf(ERROR "E35: Argument of tanh not set, NULL encountered\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  y->grad += x->grad / (cosh(y->data) * cosh(y->data));
}

// null function
void doNothing(Value *x) { return; }

void printValue(Value *x) { printf("data: %f\ngrad: %f\n", x->data, x->grad); }

void Destroy(Value **x) {
  if (x == NULL)
    return;
  if (*x == NULL) {
    x = NULL;
    return;
  }
  free(*x);
  x = NULL;
}
// NEURON
Neuron *createNeuron(size_t sz, actFunc act) {
  Neuron *neuron = malloc(sizeof(Neuron));
  if (neuron == NULL) {
#ifdef DEBUG
    printf(ERROR "E36: Unable to create neuron\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  neuron->size = sz;
  neuron->activation = act;
  neuron->bias = doubleToValue((double)rand() / (double)RAND_MAX, true);
  neuron->weights = malloc(sizeof(Value *) * sz);
  if (neuron->weights == NULL) {
#ifdef DEBUG
    printf(ERROR "E37: Unable to allocate space for weights\n" RESET);
#endif
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
#ifdef DEBUG
    printf(ERROR "E38: Unable to allocate space for intermediate\n" RESET);
#endif
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
  else if (func == _tanh)
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
#ifdef DEBUG
    printf(ERROR "E39: Unable to allocate memory for layer\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->num_of_neurons = num_of_outputs;
  out->size_of_neurons = num_of_inputs;
  out->neurons = malloc(sizeof(Neuron *) * num_of_outputs);
  if (out->neurons == NULL) {
#ifdef DEBUG
    printf(ERROR "E40: Unable to allocate space for neurons in Layer\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < num_of_outputs; i++) {
    out->neurons[i] = createNeuron(num_of_inputs, act);
  }
  return out;
}

Value **setLayer(Layer *layer, Value **inputs) {
  Value **out = malloc(sizeof(Value *) * layer->num_of_neurons);
  if (out == NULL) {
#ifdef DEBUG
    printf(ERROR "E41: Unable to create output for layer\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
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
#ifdef DEBUG
    printf(ERROR "E42: Unable to allocate space for MLP\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->num_of_inputs = num_of_inputs;
  out->num_of_layers = num_of_layers;
  out->num_of_outputs = num_of_outputs;
  out->layers = malloc(sizeof(Layer *) * num_of_layers);
  if (out->layers == NULL) {
#ifdef DEBUG
    printf(ERROR "E43: Unable to allocate space for layers in MLP\n" RESET);
#endif
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

ValueList *CreateValueList() {
  ValueList *out = malloc(sizeof(ValueList));
  if (out == NULL) {
#ifdef DEBUG
    printf(ERROR "E44: Unable to create ValueList\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->values = malloc(sizeof(Value *));
  if (out->values == NULL) {
#ifdef DEBUG
    printf(ERROR
           "E45: Unable to allocate space for elements of ValueList\n" RESET);
#endif
    exit(EXIT_FAILURE);
  }
  out->size = 0;
  out->_cap = 1;
  return out;
}
void appendValue(ValueList *lst, Value *val) {
  lst->size++;
  if (lst->size > lst->_cap) {
    while (lst->size > lst->_cap)
      lst->_cap *= 2;
    lst->values = realloc(lst->values, sizeof(Value *) * lst->_cap);
  }
  lst->values[lst->size - 1] = val;
}
Value *getValueAt(ValueList *lst, size_t index) {
  if (index >= lst->size)
    return NULL;
  return lst->values[index];
}

void topoSortList(Value *val, ValueList *lst) {
  if (val == NULL)
    return;
  for (size_t i = 0; i < val->_prevsz; i++) {
    topoSortList(val->_prev[i], lst);
  }
  appendValue(lst, val);
}

void forward(ValueList *lst) {
  Value *tmp;
  for (size_t i = 0; i < lst->size; i++) {
    tmp = lst->values[i];
    tmp->_forward(tmp);
  }
}
void backward(ValueList *lst) {
  Value *tmp;
  for (size_t i = lst->size; i > 0; i--) {
    tmp = lst->values[i - 1];
    tmp->_backward(tmp);
  }
}

void gradientDescent(ValueList *lst, double learningRate) {
  for (size_t i = 0; i < lst->size; i++) {
    if (lst->values[i]->_modifiable) {
      lst->values[i]->data -= learningRate * lst->values[i]->grad;
    }
  }
}
#endif
#endif
