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
#endif
