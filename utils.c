#include "utils.h"
#include "autoGrad.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
ValueList *CreateValueList() {
  ValueList *out = malloc(sizeof(ValueList));
  out->values = malloc(sizeof(Value *));
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
