#include "parser.h"
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
