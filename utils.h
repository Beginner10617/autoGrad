#ifndef UTILS
#define UTILS
#include "autoGrad.h"
#include <stddef.h>
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
#endif
