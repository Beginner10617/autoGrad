#include "utils.h"
#include "neuron.h"
#include "neuron_utils.h"
#include "stdio.h"

Value **labelToValueArray(int x) {
  if (x < 0 || x > 9)
    return NULL;
  Value **out = malloc(sizeof(Value *) * 10);
  for (int i = 0; i < 10; i++) {
    if (i == x)
      out[i] = createNewValue(1.0f, "x");
    else
      out[i] = createNewValue(0.0f, "x");
    sprintf(out[i]->name, "%d", i);
    out[i]->_isparameter = 1;
  }
  return out;
}

Value **imgDataToValueArray(uint8_t *img, int img_sz) {
  // convert image data [0,255] into an array
  // of same size with values [-1.0f, +1.0f]
  // Use linear scaling
  Value **out = malloc(sizeof(Value *) * img_sz);
  for (int i = 0; i < img_sz; i++) {
    out[i] = createNewValue(img[i] / 127.5f - 1, "x");
    sprintf(out[i]->name, "%d", i);
    out[i]->_isparameter = 1;
  }
  return out;
}

void freeMLP(MLP *mlp) {
  if (mlp == NULL)
    return;
  if (mlp->actfunc)
    free(mlp->actfunc);
  for (int i = 0; i < mlp->num_of_layers; i++) {
    Layer *layer = mlp->layers[i];
    if (layer == NULL)
      continue;
    for (int j = 0; j < layer->num_of_neurons; j++) {
      Neuron *neuron = layer->neurons[j];
      if (neuron == NULL)
        continue;
      if (neuron->bias)
        free(neuron->bias);
      for (int k = 0; k < neuron->dimension; k++) {
        Value *weight = neuron->weights[k];
        if (weight)
          free(weight);
      }
      free(neuron);
    }
    free(layer);
  }
  free(mlp);
}
