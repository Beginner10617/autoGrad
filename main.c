#include "autoGrad.h"
#include "stdio.h"
#include <stdbool.h>

void printValue(Value *x) {
  if (x == NULL) {
    printf("NULL passed to printValue\n");
    exit(EXIT_FAILURE);
  }
  printf("data: %f\n", x->data);
  printf("grad: %f\n", x->grad);
}
int main() {
  Value *X = doubleToValue(0.9, true);
  Value *Y = doubleToValue(-0.2, true);
  Value *Z = doubleToValue(1.1, true);
  Value *A = EmptyValue(false);
  Value *B = EmptyValue(false);
  Value *C = EmptyValue(false);

  // A = X * Y
  setMul(A, X, Y);
  // B = Z * Z
  setMul(B, Z, Z);
  // C = A + B
  setAdd(C, A, B);

  // Forward pass
  X->_forward(X);
  Y->_forward(Y);
  Z->_forward(Z);
  A->_forward(A);
  B->_forward(B);
  C->_forward(C);

  // Backward pass
  C->grad = 1;
  C->_backward(C);
  B->_backward(B);
  A->_backward(A);
  Z->_backward(Z);
  Y->_backward(Y);
  X->_backward(X);

  // Printing Values
  printf("Value X:\n");
  printValue(X);
  printf("\nValue Y:\n");
  printValue(Y);
  printf("\nValue Z:\n");
  printValue(Z);
  printf("\nValue A:\n");
  printValue(A);
  printf("\nValue B:\n");
  printValue(B);
  printf("\nValue C:\n");
  printValue(C);
  return 0;
}
