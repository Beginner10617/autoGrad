// test.c for testing the model
#include "parser.h"
#include <math.h>
#include <stdlib.h>
#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"

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

void test() {
  FILE *fptrimg, *fptrlabel;
  MLP *mlp = loadMLP("model.agm");
  if (mlp == NULL) {
    printf("Unable to open file model.agm\n");
    return;
  }

  fptrimg = fopen("dataset/t10k-images-idx3-ubyte", "rb");
  fptrlabel = fopen("dataset/t10k-labels-idx1-ubyte", "rb");

  if (fptrimg == NULL || fptrlabel == NULL) {
    printf("Error opening file!\n");
    return;
  }

  // loading dataset header
  printf("loading dataset header\n");
  uint32_t imgheader[4], labelheader[2];
  readImgHeader(imgheader, fptrimg);
  readLabelHeader(labelheader, fptrlabel);
  int image_size = imgheader[2] * imgheader[3];

  // set tree
  printf("setting up tree\n");
  Value **input = malloc(sizeof(Value *) * image_size);
  for (size_t i = 0; i < image_size; i++)
    input[i] = EmptyValue(false);
  Value **output = setMLP(mlp, input);
  Value *fake_op = sum_value_vec(output, 10);
  //
  ValueList *lst = CreateValueList();
  topoSortList(fake_op, lst);

  // loading data from the dataset
  printf("loading dataset\n");
  uint8_t label = 0;
  uint8_t *image = malloc(sizeof(uint8_t) * image_size);
  size_t correct = 0;

  for (int i = 0; i < imgheader[1]; i++) {
    if (fread(&label, 1, 1, fptrlabel) != 1) {
      printf("Error reading label\n");
      break;
    }
    if (!readNextImage(image, image_size, fptrimg))
      break;
    for (size_t px = 0; px < image_size; px++) {
      input[px]->data = image[px] / 127.5 - 1.0;
    }
    forward(lst);
    int dig = -1;
    double act = -INFINITY;
    for (size_t num = 0; num < 10; num++) {
      if (act < output[num]->data) {
        act = output[num]->data;
        dig = num;
      }
    }
    if (label == dig)
      correct++;
  }

  fclose(fptrimg);
  fclose(fptrlabel);
  DestroyGraph(&lst);
  DestroyMLP(&mlp);
  printf("%zu correct out of %u cases %.3f%% accuracy\n", correct, imgheader[1],
         correct * 100.0 / imgheader[1]);
}
int main() {
  test();
  return 0;
}
