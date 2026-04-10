#include "neuron_utils.h"
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"

Value* createNewValue(float d, char *n){
	Value *z = malloc(sizeof(Value));
	z->data = d;
	for(int i=0; i<2; i++)
		z->_prev[i] = NULL;
	z->op = ' ';
	strncpy(z->name, n, 4);
	z->name[5] = '\0';
	z->grad = 0;
	z->_backward = doNothing;
	z->visited = 0;
	z->_isparameter = 0;
	return z;
}

void doNothing(Value *x){}

Value* add(Value *x, Value *y){
	Value *z = malloc(sizeof(Value));
	z->data = x->data + y->data;
	z->_prev[0] = x;
        z->_prev[1] = y;
	z->grad = 0.0f;
	z->op = '+';
	z->_backward = addBack;	
	z->_isparameter = 0;
	return z;
}

Value* sum(Value** x, int sz){
	if(sz==0) return NULL;
	Value* z = x[0];
	for(int i=1; i<sz; i++){
		z = add(x[i], z);
	}
	return z;
}

Value* sub(Value *x, Value *y){
	Value *z = malloc(sizeof(Value));
	z->data = x->data - y->data;
	z->_prev[0] = x;
	z->_prev[1] = y;
	z->grad = 0.0f;
	z->op = '-';
	z->_backward = subBack;
	z->_isparameter = 0;
	return z;
}

void addBack(Value *z){
	Value *x = z->_prev[0];
	Value *y = z->_prev[1];
	x->grad += z->grad;
	y->grad += z->grad;
}

void subBack(Value* z){
	Value *x = z->_prev[0];
	Value *y = z->_prev[1];
	x->grad += z->grad;
	y->grad -= z->grad;
}

Value* mul(Value *x, Value *y){
	Value *z = malloc(sizeof(Value));
	z->data = x->data * y->data;
	z->_prev[0] = x; 
	z->_prev[1] = y;
	z->grad = 0.0f;
	z->op = '*';
	z->_backward = mulBack;
	z->_isparameter = 0;
	return z;
}

void mulBack(Value *z){
	Value *x = z->_prev[0];
	Value *y = z->_prev[1];
	x->grad += y->data * z->grad;
	y->grad += x->data * z->grad;	
}

Value* vtanh(Value *x){
	Value *z = malloc(sizeof(Value));
	z->data = tanh(x->data);
	z->_prev[0] = x;
	z->_prev[1] = NULL;
	z->grad = 0.0f;
	z->op = 't';
	z->_backward = tanhBack;
	z->_isparameter = 0;
	return z;
}

void tanhBack(Value* z){
	Value *x = z->_prev[0];
	x->grad += (1 - z->data * z->data) * z->grad;
}

void printv(Value *x){
	printf("%s", x->name);
	if(x->_prev[1] == NULL && x->_prev[0] == NULL){
		printf(" = %f\n", x->data);
		return;
	}
	else if(x->_prev[1] == NULL){
		printf(" = %f = %c(%s)\n", x->data, x->op, x->_prev[0]->name);
		return;
	}
	printf(" = %f = %s %c %s\n", x->data, x->_prev[0]->name, x->op, x->_prev[1]->name);
}

void backPropagate(Value* x){
	x->grad = 1.0f;
	NodeList* topo = createNodeList();
	buildTopoSort(topo, x);
	Value* node;
	for(int i=topo->size-1; i>=0; i--){
		node = topo->values[i];
	//	printf("calling back() on %s\n", node->name);
		node->_backward(node);
	}
	resetVisited(topo);
	freeNodeList(topo);
}

// For topo-sort
NodeList* createNodeList(){
	NodeList* nodelist = malloc(sizeof(NodeList));
	nodelist->cap = 8;
	nodelist->size= 0;
	nodelist->values = malloc(nodelist->cap * sizeof(Value*));
	return nodelist;
}

void freeNodeList(NodeList* nodelist){
	free(nodelist->values);
	free(nodelist);
}

void append(NodeList* nodelist, Value* newValue){
	if(nodelist->size >= nodelist->cap){
		nodelist->cap *= 2;
		nodelist->values = realloc(nodelist->values,
				nodelist->cap * sizeof(Value*));
	}
	nodelist->values[nodelist->size] = newValue;
	nodelist->size++;
}

void buildTopoSort(NodeList *nodelist, Value* v){
	if(v==NULL || v->visited>0)
		return;
	v->visited = 1;
	for(int i=0; i<2; i++)
		buildTopoSort(nodelist, v->_prev[i]);
	append(nodelist, v);
}

void resetVisited(NodeList* topo) {
    for (int i = 0; i < topo->size; i++)
        topo->values[i]->visited = 0;
}

void gradientDescentValue(Value* v, float stepSize){
	v->data -= stepSize * v->grad;
	v->grad = 0.0f;
}

Value** addValueArr(Value** x, Value** y, int sz){
	Value** z = malloc(sizeof(Value*) * sz);
	for(int i=0; i<sz; i++)
		z[i] = add(x[i], y[i]);
	return z;
}

Value** subValueArr(Value** x, Value** y, int sz){
	Value** z = malloc(sizeof(Value*) * sz);
	for(int i=0; i<sz; i++)
		z[i] = sub(x[i], y[i]);
	return z;
}

Value** sqValueArr(Value** x, int sz){
	Value** z = malloc(sizeof(Value*) * sz);
	for(int i=0; i<sz; i++)
		z[i] = mul(x[i], x[i]);
	return z;
}
