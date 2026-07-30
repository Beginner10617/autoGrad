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

Value **labelToValueArray(int x) {
  if (x < 0 || x > 9)
    return NULL;
  Value **out = malloc(sizeof(Value *) * 10);
  for (int i = 0; i < 10; i++) {
    if (i == x)
      out[i] = doubleToValue(1.0, false);
    else
      out[i] = doubleToValue(0.0f, false);
  }
  return out;
};

Value **imgDataToValueArray(uint8_t *img, int img_sz) {
  // convert image data [0,255] into an array
  // of same size with values [-1.0f, 1.0f]
  // Use linear scaling
  Value **out = malloc(sizeof(Value *) * img_sz);
  for (int i = 0; i < img_sz; i++) {
    out[i] = doubleToValue(img[i] / 127.5 - 1, false);
  }
  return out;
}

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
  /*
  for (int i = 0; i < z->_prevcap; i++) {
    printf("sum arg : %p\n", z->_prev[i]);
  }
  int y;
  scanf("%d", &y);
  */
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
  MLP *mlp = loadMLP("model.mod");
  actFunc tanh[] = {_tanh, _tanh, none};
  size_t outputs[] = {32, 16, 10};
  if (mlp == NULL) {
    printf("model.mod not found, creating new...\n");
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

      currLoss = loss->data;
      /*
            printf("\nLast layer, first neuron, first weight data and grad resp:
         %f "
                   "%f\n",
                   mlp->layers[2]->neurons[0]->weights[0]->data,
                   mlp->layers[2]->neurons[0]->weights[0]->grad);
            printf("loss = %f %f\n", currLoss, loss->grad);
            printf("output = [");
            for (int o = 0; o < 10; o++)
              printf("%f, ", predn_matrix[0][o]->data);
            printf("]\n");
            printf("ground = [");
            for (int o = 0; o < 10; o++)
              printf("%f, ", ground_truth[0][o]->data);
            printf("]\n");
            printf("delta  = [");
            for (int o = 0; o < 10; o++)
              printf("%f, ", error_delta[0][o]->data);
            printf("]\n");
            printf("sq_err = [");
            for (int o = 0; o < 10; o++)
              printf("%f, ", sq_error_delta[0][o]->data);
            printf("]\n");
            printf("devn   = [");
            for (int o = 0; o < batch_size; o++)
              printf("%f, ", devn[o]->data);
            printf("]\n");
            int ruf;
            scanf("%d", &ruf);
      */
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
    saveMLP(mlp, "model.mod");
    printf(" saved model!\n");
  }
}

int main() {
  srand((unsigned int)time(NULL));
  train(10, 0.005);
  return 0;
}
