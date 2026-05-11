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
  Value *X1 = doubleToValue(0.9, true);
  Value *X2 = doubleToValue(-0.2, true);
  Value *X3 = doubleToValue(1.1, true);
  Value *X4 = doubleToValue(0.8f, true);
  Value *X5 = doubleToValue(0.2, true);
  Value *X6 = doubleToValue(1.2, true);
  Value *X7 = doubleToValue(-0.6, true);
  Value *X8 = doubleToValue(0.7, true);
  Value *X9 = doubleToValue(-0.9, true);
  Value *Y1 = EmptyValue(false);
  Value *Y2 = EmptyValue(false);
  Value *Y3 = EmptyValue(false);
  Value *Z = EmptyValue(false);

  setSum(Y1, 3);
  addToSum(Y1, X1);
  addToSum(Y1, X2);
  addToSum(Y1, X3);
  setSum(Y2, 3);
  addToSum(Y2, X4);
  addToSum(Y2, X5);
  addToSum(Y2, X6);
  setSum(Y3, 3);
  addToSum(Y3, X7);
  addToSum(Y3, X8);
  addToSum(Y3, X9);
  setSum(Z, 3);
  addToSum(Z, Y1);
  addToSum(Z, Y2);
  addToSum(Z, Y3);

  // _forward
  Y1->_forward(Y1);
  Y2->_forward(Y2);
  Y3->_forward(Y3);
  Z->_forward(Z);

  // _backward
  Z->grad = 1.0f;
  Z->_backward(Z);
  Y3->_backward(Y3);
  Y2->_backward(Y2);
  Y1->_backward(Y1);

  printValue(X1);
  printValue(X2);
  printValue(X3);
  printf("\n");
  printValue(X4);
  printValue(X5);
  printValue(X6);
  printf("\n");
  printValue(X7);
  printValue(X8);
  printValue(X9);
  printf("\n");
  printValue(Y1);
  printValue(Y2);
  printValue(Y3);
  printf("\n");
  printValue(Z);

  return 0;
}
