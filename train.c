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
  struct timespec start, end;

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
  double forward_dt = 0, backward_dt = 0, gd_dt = 0, input_load = 0,
         debug_dt = 0, saving_dt = 0;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int iter = 0; iter < iterations; iter++) {
    for (int j = 0; j < num_of_batches; j++) {
      struct timespec s, e;
      int off = j * batch_size;

      clock_gettime(CLOCK_MONOTONIC, &s);
      for (int ipt = 0; ipt < batch_size && ipt + off < imgheader[1]; ipt++) {
        for (int px = 0; px < image_size; px++)
          input_matrix[ipt][px]->data =
              ((double)all_images[ipt + off][px]) / 127.5 - 1.0;

        for (int dig = 0; dig < 10; dig++)
          ground_truth[ipt][dig]->data = truth[ipt + off] == dig ? 1.0 : 0.0;
      }
      clock_gettime(CLOCK_MONOTONIC, &e);
      input_load += (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC, &s);
      forward(val_lst);
      clock_gettime(CLOCK_MONOTONIC, &e);
      forward_dt += (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC, &s);
      backward(val_lst);
      clock_gettime(CLOCK_MONOTONIC, &e);
      backward_dt += (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC, &s);
      gradientDescent(val_lst, stepSize);
      clock_gettime(CLOCK_MONOTONIC, &e);
      gd_dt += (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC, &s);
      /*for (size_t i = 0; i < val_lst->size; i++) {
        if (val_lst->values[i]->_modifiable) {
          max_grad = max_grad > fabs(val_lst->values[i]->grad)
                         ? max_grad
                         : fabs(val_lst->values[i]->grad);
          max_param = max_param > fabs(val_lst->values[i]->data)
                          ? max_param
                          : fabs(val_lst->values[i]->data);
        }
      }*/
      currLoss = loss->data;

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
      clock_gettime(CLOCK_MONOTONIC, &e);
      debug_dt += (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;
    }
    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);
    saveMLP(mlp, "model.agm");
    clock_gettime(CLOCK_MONOTONIC, &e);
    saving_dt += (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) / 1e9;

    printf(" saved model!\n");
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double tot_time =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  double sum =
      forward_dt + backward_dt + gd_dt + debug_dt + saving_dt + input_load;
  DestroyGraph(&val_lst);
  DestroyMLP(&mlp);
  for (int i = 0; i < imgheader[1]; i++)
    free(all_images[i]);
  free(all_images);
  all_images = NULL;
  for (int i = 0; i < 80; i++)
    printf("-");
  printf("\nTIME ANALYSIS\n");
  printf("Input loading:         %.2f%%\n", input_load * 100 / tot_time);
  printf("Forward propagation:   %.2f%%\n", forward_dt * 100 / tot_time);
  printf("Backward propagation:  %.2f%%\n", backward_dt * 100 / tot_time);
  printf("Gradient descent:      %.2f%%\n", gd_dt * 100 / tot_time);
  printf("I/O time:              %.2f%%\n", debug_dt * 100 / tot_time);
  printf("Saving mlp parameters: %.2f%%\n", saving_dt * 100 / tot_time);
  printf("Others:                %.2f%%\n", (tot_time - sum) * 100 / tot_time);
  printf("Total time:            %.2fs\n", tot_time);
}

int main() {
  srand((unsigned int)time(NULL));
  train(10, 0.0005);
  return 0;
}
