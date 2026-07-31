#include "parser.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define AUTOGRAD_IMPLEMENTATION
#define DEBUG
#include "autograd.h"
#define BAR_WIDTH 50 // For progress bar

Value **subValueArr(Value **x, Value **y, size_t sz) {
  if (sz == 0)
    return NULL;
  Value **z = malloc(sizeof(Value *) * sz);
  for (int i = 0; i < sz; i++) {
    z[i] = EmptyValue(false);
    setSub(z[i], x[i], y[i]);
  }
  return z;
}

Value **sqValueArr(Value **x, size_t sz) {
  if (sz == 0)
    return NULL;
  Value **z = malloc(sizeof(Value *) * sz);
  for (int i = 0; i < sz; i++) {
    z[i] = EmptyValue(false);
    setMul(z[i], x[i], x[i]);
  }
  return z;
}

Value *sum_value_vec(Value **x, int sz) {
  if (sz == 0)
    return NULL;
  Value *z = EmptyValue(false);
  setSum(z, sz);
  for (int i = 0; i < sz; i++) {
    addToSum(z, x[i]);
  }
  return z;
}

void train(int iterations, double stepSize) {

  FILE *fptrimg, *fptrlabel;

  fptrimg = fopen("dataset/train-images-idx3-ubyte", "rb");
  fptrlabel = fopen("dataset/train-labels-idx1-ubyte", "rb");

  if (fptrimg == NULL || fptrlabel == NULL) {
    printf("Error opening file!\n");
    return;
  }

  // loading dataset header
  uint32_t imgheader[4], labelheader[2];
  readImgHeader(imgheader, fptrimg);
  readLabelHeader(labelheader, fptrlabel);

  // loading data from the dataset
  int image_size = imgheader[2] * imgheader[3];
  uint8_t label = 0;
  uint8_t truth[labelheader[1]];
  uint8_t **all_images = malloc(sizeof(uint8_t *) * imgheader[1]);

  for (int i = 0; i < imgheader[1]; i++) {
    if (fread(&label, 1, 1, fptrlabel) != 1) {
      printf("Error reading label\n");
      break;
    }
    all_images[i] = malloc(sizeof(uint8_t) * image_size);
    if (!readNextImage(all_images[i], image_size, fptrimg))
      break;
    truth[i] = label;
  }

  fclose(fptrimg);
  fclose(fptrlabel);
  printf("Image data loaded!\n");
  printf("Opening MLP...\n");
  MLP *mlp = loadMLP("model.agm");
  actFunc tanh[] = {_tanh, _tanh, none};
  size_t outputs[] = {32, 16, 10};
  if (mlp == NULL) {
    printf("model.agm not found, creating new...\n");
    mlp = createMLP(3, image_size, outputs, tanh);
  }

  printf("Allocating memory for training data...\n");

  int batch_size = 10;
  float currLoss;

  printf("Starting training loop...\n");
  int num_of_batches = imgheader[1] / batch_size;

  // set the tree
  Value ***input_matrix = malloc(sizeof(Value **) * batch_size);
  Value ***predn_matrix = malloc(sizeof(Value **) * batch_size);
  Value ***error_delta = malloc(sizeof(Value **) * batch_size);
  Value ***sq_error_delta = malloc(sizeof(Value **) * batch_size);
  Value **devn = malloc(sizeof(Value *) * batch_size);
  Value ***ground_truth = malloc(sizeof(Value **) * batch_size);
  Value ***img_inputs = malloc(sizeof(Value **) * batch_size);
  Value *loss;
  for (int i = 0; i < batch_size; i++) {
    input_matrix[i] = malloc(sizeof(Value *) * image_size);
    for (int j = 0; j < image_size; j++) {
      input_matrix[i][j] = EmptyValue(false);
    }
    predn_matrix[i] = setMLP(mlp, input_matrix[i]);
    ground_truth[i] = malloc(sizeof(Value *) * 10);

    for (int dig = 0; dig < 10; dig++)
      ground_truth[i][dig] = EmptyValue(false);

    error_delta[i] = subValueArr(ground_truth[i], predn_matrix[i], 10);
    sq_error_delta[i] = sqValueArr(error_delta[i], 10);
    devn[i] = sum_value_vec(sq_error_delta[i], 10);
  }
  loss = sum_value_vec(devn, batch_size);
  //
  ValueList *val_lst = CreateValueList();
  topoSortList(loss, val_lst);
  printf("Computation tree created!\n");
  printf("Size of value list : %zu\n", val_lst->size);

  // training
  for (int iter = 0; iter < iterations; iter++) {
    double max_grad = 0, max_param = 0, max_loss = 0;
    for (int j = 0; j < num_of_batches; j++) {
      int off = j * batch_size;

      for (int ipt = 0; ipt < batch_size && ipt + off < imgheader[1]; ipt++) {
        for (int px = 0; px < image_size; px++)
          input_matrix[ipt][px]->data =
              ((double)all_images[ipt + off][px]) / 127.5 - 1.0;

        for (int dig = 0; dig < 10; dig++)
          ground_truth[ipt][dig]->data = truth[ipt + off] == dig ? 1.0 : 0.0;
      }

      forward(val_lst);
      backward(val_lst);
      gradientDescent(val_lst, stepSize);

      for (size_t i = 0; i < val_lst->size; i++) {
        if (val_lst->values[i]->_modifiable) {
          max_grad = max_grad > fabs(val_lst->values[i]->grad)
                         ? max_grad
                         : fabs(val_lst->values[i]->grad);
          max_param = max_param > fabs(val_lst->values[i]->data)
                          ? max_param
                          : fabs(val_lst->values[i]->data);
        }
      }
      currLoss = loss->data;
      max_loss = max_loss > currLoss ? max_loss : currLoss;

     // progress bar
      printf("\r[");
      int pos = ((j + 1) * BAR_WIDTH) / (num_of_batches);

      for (int i = 0; i < BAR_WIDTH; i++) {
        if (i < pos)
          printf("=");
        else if (i == pos)
          printf(">");
        else
          printf(" ");
      }

      printf("] iteration %d/%d batch %d/%d loss = %f", iter + 1, iterations,
             j + 1, num_of_batches, currLoss);

      fflush(stdout);
    }
    printf("\nIteration: %d\nMax Loss: %f\nMax |grad|: %f\nMax |param|: %f\n", iter,
           max_loss, max_grad, max_param);
    saveMLP(mlp, "model.agm");
    printf("saved model!\n");
  }
  DestroyGraph(&val_lst);
  DestroyMLP(&mlp);
}

int main() {
  srand((unsigned int)time(NULL));
  train(10, 0.0005);
  return 0;
}
