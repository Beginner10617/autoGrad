#ifndef AUTOGRAD
#define AUTOGRAD
#include "stdbool.h"
#include "stdlib.h"
typedef struct Value Value;
typedef void (*Funcptr)(Value *);

struct Value {
  double data, grad;
  Funcptr _backward, _forward;
  struct Value **_prev;
  int _prevsz, _prevcap;
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

// _forward
void _addFwd(Value *x);
void _subFwd(Value *x);
void _mulFwd(Value *x);

// _backward
void _addBack(Value *x);
void _subBack(Value *x);
void _mulBack(Value *x);

// null function
void doNothing(Value *x);

// Destructors
void Destroy(Value **x);
#endif
