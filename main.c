#include "parser.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#define AUTOGRAD_IMPLEMENTATION
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
    setMul(z[i], x[i], x[i]);
  }
  return z;
}

Value *sum_value_vec(Value **x, int sz) {
  if (sz == 0)
    return NULL;
  Value *z = EmptyValue(false);
  setSum(z, sz);
  for (int i = 1; i < sz; i++) {
    addToSum(z, x[i]);
  }
  return z;
}

void train(int iterations, float stepSize) {

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
  uint8_t all_images[imgheader[1]][image_size];

  for (int i = 0; i < imgheader[1]; i++) {
    if (fread(&label, 1, 1, fptrlabel) != 1) {
      printf("Error reading label\n");
      break;
    }
    if (!readNextImage(all_images[i], image_size, fptrimg))
      break;
    truth[i] = label;
  }

  fclose(fptrimg);
  fclose(fptrlabel);
  printf("Image data loaded!\n");
  printf("Opening MLP...\n");
  MLP *mlp = loadMLP("model.txt");
  actFunc tanh[] = {_tanh, _tanh, none};
  size_t outputs[] = {32, 16, 10};
  if (mlp == NULL) {
    printf("model not found, creating new...\n");
    mlp = createMLP(3, image_size, outputs, tanh);
  }

  printf("Allocating memory for training data...\n");

  int batch_size = 10;
  /*
    Value ***ypred = malloc(sizeof(Value **) * batch_size);
    Value ***dely = malloc(sizeof(Value **) * batch_size);
    Value ***sqdely = malloc(sizeof(Value **) * batch_size);
    Value **devn = malloc(sizeof(Value *) * batch_size);
    Value *losssum;
  */
  float currLoss;

  printf("Starting training loop...\n");
  int asd, num_of_batches = imgheader[1] / batch_size;

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
    error_delta[i] = subValueArr(ground_truth[i], predn_matrix[i], 10);
    sq_error_delta[i] = sqValueArr(error_delta[i], 10);
    devn[i] = sum_value_vec(sq_error_delta[i], 10);
  }
  loss = sum_value_vec(devn, batch_size);
  //

  // training
  for (int iter = 0; iter < iterations; iter++) {
    for (int j = 0; j < num_of_batches; j++) {
      int off = j * batch_size;
      /*
      for (int ipt = 0; ipt < batch_size && ipt + off < imgheader[1]; ipt++) {
        ypred[ipt] = evaluateMLP(mlp, img_inputs[ipt + off]);
        dely[ipt] = subValueArr(ground_truth[ipt + off], ypred[ipt], 10);
        sqdely[ipt] = sqValueArr(dely[ipt], 10);
        devn[ipt] = sum(sqdely[ipt], 10);
      }
      losssum = sum(devn, batch_size);

      loss itself would be the output of the tree
      */
      currLoss = losssum->data;
      backPropagate(losssum);
      gradientDescentMLP(mlp, stepSize);

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
    saveMLP(mlp, "model.txt");
    printf(" saved model!\n");
  }
}

int main() {
  srand((unsigned int)time(NULL));
  train(10, 0.005f);
  return 0;
}
