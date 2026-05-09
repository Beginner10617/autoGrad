#include "autoGrad.h"
#include "stdio.h"
#include <stdlib.h>
// Constructors
Value *EmptyValue() {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
    printf("Error allocating space for value\n");
    exit(EXIT_FAILURE);
  }
  out->data = 0;
  out->grad = 0;
  out->_backward = doNothing;
  out->_forward = doNothing;
  out->_prev[0] = NULL;
  out->_prev[1] = NULL;
  out->_next = NULL;
  out->nextSize = 0;
  out->nextCapacity = 1;
  return out;
}

Value *floatToValue(float x) {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
    printf("Error allocating space for value\n");
    exit(EXIT_FAILURE);
  }
  out->data = x;
  out->grad = 0;
  out->_backward = doNothing;
  out->_forward = doNothing;
  out->_prev[0] = NULL;
  out->_prev[1] = NULL;
  out->_next = NULL;
  out->nextSize = 0;
  out->nextCapacity = 1;
  return out;
}

Value *doubleToValue(double x) {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
    printf("Error allocating space for value\n");
    exit(EXIT_FAILURE);
  }
  out->data = x;
  out->grad = 0;
  out->_backward = doNothing;
  out->_forward = doNothing;
  out->_prev[0] = NULL;
  out->_prev[1] = NULL;
  out->_next = NULL;
  out->nextSize = 0;
  out->nextCapacity = 1;
  return out;
}

// helper-1
void addNextValue(Value *x, Value *out) {
  if (x == NULL || out == NULL) {
    printf("NULL passed to addNextValue\n");
    exit(EXIT_FAILURE);
  }
  x->nextSize += 1;
  if (x->nextSize > x->nextCapacity) {
    while (x->nextSize > x->nextCapacity) {
      x->nextCapacity *= 2;
    }
    x->_next = realloc(x->_next, sizeof(Value *) * x->nextCapacity);
  }
  x->_next[x->nextSize - 1] = out;
}

// set out = x <op> y
void setAdd(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
    printf("NULL passed to setAdd\n");
    exit(EXIT_FAILURE);
  }
  out->_forward = _addFwd;
  out->_backward = _addBack;
  out->_prev[0] = x;
  out->_prev[1] = y;

  addNextValue(x, out);
  addNextValue(y, out);
}

void setSub(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
    printf("NULL passed to setAdd\n");
    exit(EXIT_FAILURE);
  }
  out->_forward = _subFwd;
  out->_backward = _subBack;
  out->_prev[0] = x;
  out->_prev[1] = y;

  addNextValue(x, out);
  addNextValue(y, out);
}

void setMul(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
    printf("NULL passed to setAdd\n");
    exit(EXIT_FAILURE);
  }
  out->_forward = _mulFwd;
  out->_backward = _mulBack;
  out->_prev[0] = x;
  out->_prev[1] = y;

  addNextValue(x, out);
  addNextValue(y, out);
}

// _forward : evaluate out = x <op> y
void _addFwd(Value *x) {
  if (x == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data + x->_prev[1]->data;
}

void _subFwd(Value *x) {
  if (x == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data - x->_prev[1]->data;
}

void _mulFwd(Value *x) {
  if (x == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data * x->_prev[1]->data;
}

// _backward
