#include "parser.h"
#include "autograd.h"
#include <stdbool.h>
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

const char *error_messages[] = {
    "",
    "Unable to open file\n",
    "EOF while reading num_of_inputs\n",
    "EOF while reading num_of_layers\n",
    "EOF while reading num_of_outputs\n",
    "EOF while reading layer_activation\n",
    "EOF while reading neuron_parameter\n",
    "No EOF after completion of data\n",
};

void saveMLP(MLP *mlp, const char *Fname) {
  FILE *file = fopen(Fname, "wb");
  if (file == NULL) {
#ifdef DEBUG
    printf(ERROR "Not able to write to file %s\n" RESET, Fname);
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

MLP *loadMLP(const char *Fname) {
  int err = validate(Fname);
  if (err) {
#ifdef DEBUG
    printf("%s", error_messages[err]);
#endif
    return NULL;
  }
  FILE *file = fopen(Fname, "rb");
  if (file == NULL) {
#ifdef DEBUG
    printf("Unable to open file %s from loadMLP\n", Fname);
#endif
    return NULL;
  }
  MLP *mlp = malloc(sizeof(MLP));
  if (mlp == NULL) {
#ifdef DEBUG
    printf("Unable to allocate space for mlp from loadMLP\n");
#endif
    return NULL;
  }
  fread(&mlp->num_of_inputs, sizeof(size_t), 1, file);
  fread(&mlp->num_of_layers, sizeof(size_t), 1, file);
  mlp->num_of_outputs = malloc(sizeof(size_t) * mlp->num_of_layers);
  mlp->layers = malloc(sizeof(Layer **) * mlp->num_of_layers);
  for (size_t i = 0; i < mlp->num_of_layers; i++) {
    mlp->layers[i] = malloc(sizeof(Layer));
    mlp->layers[i]->num_of_neurons = mlp->num_of_outputs[i];
    mlp->layers[i]->size_of_neurons =
        i ? mlp->num_of_outputs[i - 1] : mlp->num_of_inputs;
    fread(&mlp->layers[i]->activation, sizeof(int), 1, file);
    for (size_t j = 0; j < mlp->layers[i]->num_of_neurons; j++) {
      mlp->layers[i]->neurons[j] = malloc(sizeof(Neuron));
      mlp->layers[i]->neurons[j]->size = mlp->layers[i]->size_of_neurons;
      mlp->layers[i]->neurons[j]->activation = mlp->layers[i]->activation;
      mlp->layers[i]->neurons[j]->bias = malloc(sizeof(Value));
      mlp->layers[i]->neurons[j]->bias->_modifiable = true;
      mlp->layers[i]->neurons[j]->weights =
          malloc(sizeof(Value *) * mlp->layers[i]->size_of_neurons);
      for (size_t k = 0; k < mlp->layers[i]->size_of_neurons; k++)
        mlp->layers[i]->neurons[j]->weights[k]->_modifiable = true;
      fread(&mlp->layers[i]->neurons[j]->bias->data, sizeof(double), 1, file);
      fread(mlp->layers[i]->neurons[j]->weights, sizeof(double),
            mlp->layers[i]->size_of_neurons, file);
    }
  }
  return mlp;
}

int validate(const char *Fname) {
  FILE *file = fopen(Fname, "rb");
  if (file == NULL)
    return 1;

  size_t mlp_num_of_inputs, mlp_num_of_layers;
  int layer_activation;

  if (!fread(&mlp_num_of_inputs, sizeof(size_t), 1, file))
    return 2;

  if (!fread(&mlp_num_of_layers, sizeof(size_t), 1, file))
    return 3;

  size_t mlp_num_of_outputs[mlp_num_of_layers];
  if (fread(mlp_num_of_outputs, sizeof(size_t), mlp_num_of_layers, file) !=
      mlp_num_of_layers)
    return 4;

  for (size_t i = 0; i < mlp_num_of_layers; i++) {
    if (!fread(&layer_activation, sizeof(int), 1, file))
      return 5;

    for (size_t j = 0; j < mlp_num_of_outputs[i]; j++) {
      size_t dimension = i ? mlp_num_of_outputs[i - 1] : mlp_num_of_inputs;
      double neuron_parameter[dimension + 1];
      if (fread(&neuron_parameter, sizeof(double), dimension + 1, file) !=
          dimension + 1)
        return 6;
    }
  }
  if (fgetc(file) != EOF) {
    fclose(file);
    return 7;
  }
  fclose(file);
  return 0;
}
