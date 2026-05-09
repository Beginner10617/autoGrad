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
  return out;
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
void _addBack(Value *x) {
  if (x == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  x->_prev[0]->grad += x->grad;
  x->_prev[1]->grad += x->grad;
}

void _subBack(Value *x) {
  if (x == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  x->_prev[0]->grad += x->grad;
  x->_prev[1]->grad -= x->grad;
}

void _mulBack(Value *z) {
  if (z == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  Value *x = z->_prev[0];
  Value *y = z->_prev[1];
  if (x == NULL || y == NULL) {
    printf("NULL passed to _addFwd\n");
    exit(EXIT_FAILURE);
  }
  x->grad += y->data * z->grad;
  y->grad += x->data * z->grad;
}

// null function
void doNothing(Value *x) { return; }

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
