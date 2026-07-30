#include "parser.h"
#include "autograd.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t swap(uint32_t x) {
  return ((x >> 24) & 0xff) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) |
         ((x << 24) & 0xff000000);
}

void swapInPlace(uint32_t *arr, int size) {
  for (int i = 0; i < size; i++) {
    *(arr + i) = swap(*(arr + i));
  }
  return;
}

void readImgHeader(uint32_t *head, FILE *ptr) {
  if (fread(head, 4, 4, ptr) != 4) {
    printf("Error reading the Image header\n");
    fclose(ptr);
    exit(1);
  }
  swapInPlace(head, 4);
  printf("-------------------------\n");
  printf("magic number: %d\n", head[0]);
  printf("number of images: %d\n", head[1]);
  printf("number of rows: %d\n", head[2]);
  printf("number of columns: %d\n", head[3]);
  printf("-------------------------\n");
  return;
}

void readLabelHeader(uint32_t *head, FILE *ptr) {
  if (fread(head, 4, 2, ptr) != 2) {
    printf("Error reading the Label header\n");
    fclose(ptr);
    exit(1);
  }
  swapInPlace(head, 2);
  printf("-------------------------\n");
  printf("magic number: %d\n", head[0]);
  printf("number of images: %d\n", head[1]);
  printf("-------------------------\n");
  return;
}

int readNextImage(uint8_t *img, int img_size, FILE *ptr) {
  if (fread(img, 1, img_size, ptr) != img_size) {
    if (feof(ptr))
      return 0;
    printf("Error reading images\n");
    return 0;
  }
  return 1;
}

const char *error_messages[] = {};

void saveMLP(MLP *mlp, const char *Fname) {
  FILE *file = fopen(Fname, "wb");
  if (file == NULL) {
#ifdef DEBUG
    printf(ERROR "Not able to write to file %s" RESET, Fname);
#endif
    return;
  }
  fwrite(&mlp->num_of_inputs, sizeof(size_t), 1, file);
  fwrite(&mlp->num_of_layers, sizeof(size_t), 1, file);
  fwrite(mlp->num_of_outputs, sizeof(size_t), mlp->num_of_layers, file);
  for (size_t i = 0; i < mlp->num_of_layers; i++) {
    Layer *layer = mlp->layers[i];
    fwrite(&layer->num_of_neurons, sizeof(size_t), 1, file);
    fwrite(&layer->size_of_neurons, sizeof(size_t), 1, file);
    fwrite(&layer->activation, sizeof(int), 1, file);
    for (size_t j = 0; j < layer->num_of_neurons; j++) {
      Neuron *neuron = layer->neurons[j];
      fwrite(&neuron->size, sizeof(size_t), 1, file);
      fwrite(&neuron->bias->data, sizeof(double), 1, file);
      for (size_t k = 0; k < neuron->size; k++) {
        fwrite(&neuron->weights[k]->data, sizeof(double), 1, file);
      }
    }
  }
}

MLP *loadMLP(const char *Fname) { return NULL; }

int validate(const char *Fname) { return 0; }
