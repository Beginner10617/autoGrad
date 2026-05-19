#include "autoGrad.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdlib.h>
#define ERROR "\x1b[31m"
#define RESET "\x1b[0m"
// Constructors
Value *EmptyValue(bool modify) {
  Value *out = malloc(sizeof(Value));
  if (out == NULL) {
    printf(ERROR "E01: Error allocating space for value\n" RESET);
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
    printf(ERROR "E02: Error allocating space for value\n" RESET);
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
    printf(ERROR "E03: Error allocating space for value\n" RESET);
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
    printf(ERROR "E04: NULL passed to setAdd\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_forward = _addFwd;
  out->_backward = _addBack;
  out->_prevsz = 2;
  out->_prevcap = 2;
  out->_prev = malloc(sizeof(Value *) * 2);
  if (out->_prev == NULL) {
    printf(ERROR "E05: Unable to allocate _prev inside setAdd\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = x;
  out->_prev[1] = y;
}

void setSub(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
    printf(ERROR "E06: NULL passed to setSub\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_forward = _subFwd;
  out->_backward = _subBack;
  out->_prevsz = 2;
  out->_prevcap = 2;
  out->_prev = malloc(sizeof(Value *) * 2);
  if (out->_prev == NULL) {
    printf(ERROR "E07: Unable to allocate _prev inside setSub\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = x;
  out->_prev[1] = y;
}

void setMul(Value *out, Value *x, Value *y) {
  if (out == NULL || x == NULL || y == NULL) {
    printf(ERROR "E08: NULL passed to setMul\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_forward = _mulFwd;
  out->_backward = _mulBack;
  out->_prevsz = 2;
  out->_prevcap = 2;
  out->_prev = malloc(sizeof(Value *) * 2);
  if (out->_prev == NULL) {
    printf(ERROR "E09: Unable to allocate _prev inside setMul\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = x;
  out->_prev[1] = y;
}
void setTanh(Value *out, Value *in) {
  if (out == NULL || in == NULL) {
    printf(ERROR "E10: NULL passed to setTanh\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_backward = _tanhBack;
  out->_forward = _tanhFwd;
  out->_prevsz = 1;
  out->_prevcap = 1;
  out->_prev = malloc(sizeof(Value *));
  if (out->_prev == NULL) {
    printf(ERROR "E11: Unable to allocate _prev inside setTanh\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_prev[0] = in;
}
void setSum(Value *out, size_t size) {
  if (out == NULL || size == 0) {
    printf(ERROR "E12: NULL passed to setSum\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_forward = _sumFwd;
  out->_backward = _sumBack;
  out->_prevsz = 0;
  out->_prevcap = size;
  out->_prev = malloc(sizeof(Value *) * out->_prevcap);
  if (out->_prev == NULL) {
    printf(ERROR "E13: Unable to allocate _prev inside setSum\n" RESET);
    exit(EXIT_FAILURE);
  }
}
void addToSum(Value *out, Value *x) {
  if (out->_prevsz >= out->_prevcap) {
    printf(ERROR "E14: Parameters to sum overflowed\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x == NULL) {
    printf(ERROR "E15: NULL passed to addToSum\n" RESET);
    exit(EXIT_FAILURE);
  }
  out->_prev[out->_prevsz] = x;
  out->_prevsz++;
}

// _forward : evaluate out = x <op> y
void _addFwd(Value *x) {
  if (x == NULL) {
    printf(ERROR "E16: NULL passed to _addFwd\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf(ERROR "E17: Argument of add not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data + x->_prev[1]->data;
}

void _subFwd(Value *x) {
  if (x == NULL) {
    printf(ERROR "E18: NULL passed to _subFwd\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf(ERROR "E19: Argument of sub not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data - x->_prev[1]->data;
}

void _mulFwd(Value *x) {
  if (x == NULL) {
    printf(ERROR "E20: NULL passed to _mulFwd\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf(ERROR "E21: Argument of mul not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->data = x->_prev[0]->data * x->_prev[1]->data;
}
void _sumFwd(Value *x) {
  if (x == NULL) {
    printf(ERROR "E22: NULL passed to _sumFwd\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->data = 0;
  for (size_t i = 0; i < x->_prevcap; i++) {
    if (x->_prev[i] == NULL) {
      printf(ERROR "E23: Argument of sum not set, NULL encountered\n" RESET);
      exit(EXIT_FAILURE);
    }
    x->data += x->_prev[i]->data;
  }
}
void _tanhFwd(Value *x) {
  if (x == NULL) {
    printf(ERROR "E24: NULL passed to _tanhFwd\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x->_prev == NULL) {
    printf(ERROR "E25: Argument of tanh not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->data = tanh(x->_prev[0]->data);
}

// _backward
void _addBack(Value *x) {
  if (x == NULL) {
    printf(ERROR "E26: NULL passed to _addBack\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf(ERROR "E27: Argument of add not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->_prev[0]->grad += x->grad;
  x->_prev[1]->grad += x->grad;
}

void _subBack(Value *x) {
  if (x == NULL) {
    printf(ERROR "E28: NULL passed to _subBack\n" RESET);
    exit(EXIT_FAILURE);
  }
  if (x->_prev[0] == NULL || x->_prev[1] == NULL) {
    printf(ERROR "E29: Argument of sub not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->_prev[0]->grad += x->grad;
  x->_prev[1]->grad -= x->grad;
}

void _mulBack(Value *z) {
  if (z == NULL) {
    printf(ERROR "E30: NULL passed to _mulBack\n" RESET);
    exit(EXIT_FAILURE);
  }
  Value *x = z->_prev[0];
  Value *y = z->_prev[1];
  if (x == NULL || y == NULL) {
    printf(ERROR "E31: Argument of mul not set, NULL encountered\n" RESET);
    exit(EXIT_FAILURE);
  }
  x->grad += y->data * z->grad;
  y->grad += x->data * z->grad;
}
void _sumBack(Value *x) {
  if (x == NULL) {
    printf(ERROR "E32: NULL passed to _sumBack\n" RESET);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < x->_prevcap; i++) {
    if (x->_prev[i] == NULL) {
      printf(ERROR "E33: Argument of sum not set, NULL encountered\n" RESET);
      exit(EXIT_FAILURE);
    }
    x->_prev[i]->grad += x->grad;
  }
}
void _tanhBack(Value *x) {
  if (x == NULL) {
    printf(ERROR "E34: NULL passed to _tanhBack\n" RESET);
    exit(EXIT_FAILURE);
  }
  Value *y = x->_prev[0];
  if (x == NULL || y == NULL) {
    printf(ERROR "E35: Argument of tanh not set, NULL encountered\n" RESET);
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
