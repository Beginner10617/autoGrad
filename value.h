#ifndef NEURON_UTILS_H
#define NEURON_UTILS_H
#include "stdlib.h"

typedef struct Value Value;

typedef void (*Funcptr)(Value*);

struct Value{
	float data, grad;
	struct Value* _prev[2];	// Only 2 operands
	char op;
	char name[6]; // For debugging
	Funcptr _backward;
	// For topo-sort	
	int visited;
	// Fpr computation tree
	int _isparameter;
};

// For topo-sort
typedef struct {
	Value** values;
	int size, cap;
}NodeList;

// Creating nodelist for topo-sort
NodeList* createNodeList();

// Appending elements to the nodelist
void append(NodeList* nodelist, Value* newValue);

// Building nodelist with give value as the root
void buildTopoSort(NodeList *nodelist, Value* v);

// Reset-ing visited flags after topo-sort
void resetVisited(NodeList *nodelist);

// Freeing allocated memory to a nodelist after backpropagation
void freeNodeList(NodeList* nodelist);

// Initialising a value
Value* createNewValue(float data, char* name);

// Operations on value
Value* add(Value* x, Value* y);
Value* sub(Value* x, Value* y);
Value* mul(Value* x, Value* y);
Value* vtanh(Value* x);
Value* sum(Value** x, int sz);
Value** addValueArr(Value** x, Value** y, int sz);
Value** subValueArr(Value** x, Value** y, int sz);
Value** sqValueArr(Value** x, int sz);

// For debugging
void printv(Value* x);

// For back-propagation
void addBack(Value* x);
void subBack(Value* x);
void mulBack(Value* x);
void tanhBack(Value* x);
void doNothing(Value* x);

// Back-propagation with given value as the root
void backPropagate(Value* x);

// Update data and reset grad = 0 
void gradientDescentValue(Value* v, float stepSize);

#endif
