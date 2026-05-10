#ifndef UTILS
#define UTILS
#include "autoGrad.h"
typedef struct ValueList List;

// a list to hold topo-sorted list of values, executing fwd-bwd passes on it
struct ValueList {
  Value **values;
  int size = 0, _cap = 1;
};
// helper function
void appendValue(ValueList *lst, Value *val);
// toposort
ValueList *topoSortList(Value *val);
// fwd-bwd passes, compute topoSortList once and use multiple times
void forward(ValueList *lst);
void backward(ValueList *lst);
// modifying values - gradient descent
void gradientDescent(ValueList *lst);
#endif
