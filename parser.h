#ifndef PARSER_H
#define PARSER_H
#include "autograd.h"
#include <stdint.h>
#include <stdio.h>
// Converts little-endian to big-endian
uint32_t swap(uint32_t x);
// calls swap() and replaces the output in-place
void swapInPlace(uint32_t *arr, int size);
// Reads and display head of image dataset
void readImgHeader(uint32_t *head, FILE *ptr);
// Reads and display head of label dataset
void readLabelHeader(uint32_t *head, FILE *ptr);
// Reads next image int uint8_t *img
int readNextImage(uint8_t *img, int img_size, FILE *ptr);
// Save MLP parameters in a textfile
void saveMLP(MLP *mlp, const char *Fname);
// Load MLP from parameters specified in a textfile
MLP *loadMLP(const char *Fname);
// validate the file before parsing, non zero value in case of error
int validate(const char *Fname);
#endif
