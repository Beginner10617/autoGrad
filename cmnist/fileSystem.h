#ifndef FILE_SYSTEM_H
#define FILE_SYSTEM_H
#include "neuron.h"
#include "stdio.h"

// Save MLP parameters in a textfile
void saveMLP(MLP *mlp, const char *Fname);

// Load MLP from parameters specified in a textfile
MLP *loadMLP(const char *Fname);

// validate the file before parsing, non zero value in case of error
int validate(const char *Fname);
#endif
