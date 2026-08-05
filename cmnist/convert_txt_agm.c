#include "fileSystem.h"

void saveMLP_agm(MLP *mlp, const char *Fname) {
  FILE *file = fopen(Fname, "wb");
  if(file == NULL){
    printf("Unable to write file %s", Fname);
    return;
  }
  size_t tmp; double tmp_d;
  tmp = (size_t) mlp->num_of_inputs;
  fwrite(&tmp, sizeof(size_t), 1, file);
  tmp = (size_t) mlp->num_of_layers;
  fwrite(&tmp, sizeof(size_t), 1, file);
  for (size_t i = 0; i < mlp->num_of_layers; i++) {
    tmp = (size_t) mlp->layers[i]->num_of_neurons;
    fwrite(&tmp, sizeof(size_t), 1, file);
  }
  for (size_t i = 0; i < mlp->num_of_layers; i++) {
    tmp = (size_t) mlp->actfunc[i];
    fwrite(&tmp, sizeof(size_t), 1, file);
    Layer *layer = mlp->layers[i];
    for (size_t j = 0; j < layer->num_of_neurons; j++) {
      Neuron *neuron = layer->neurons[j];
      tmp_d = (double) neuron->bias->data;
      fwrite(&tmp_d, sizeof(double), 1, file);
      for (size_t k = 0; k < neuron->dimension; k++) {
        tmp_d = (double) neuron->weights[k]->data;
        fwrite(&tmp_d, sizeof(double), 1, file);
      }
    }
  }
  fclose(file);
}

int main(int argc, char *argv[]){
  if(argc != 3){
    printf("Usage : %s <text-file> <output-file-name>\n", argv[0]);
    return 1;
  }
  MLP *mlp = loadMLP(argv[1]);
  if(mlp == NULL) return 1;
  saveMLP_agm(mlp, argv[2]);
  return 0;
}
