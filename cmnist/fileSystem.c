#include "fileSystem.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

const char *error_messages[] = {
    "Unreachable",
    "mlp names should be of 5 characters or less",
    "(mlp) invalid input, value can't be converted to positive integer",
    "(mlp) integer value wasn't supposed to be a zero",
    "layer names should be of 5 characters or less",
    "(layer) invalid input, value can't be converted to positive integer",
    "(layer) integer value wasn't supposed to be a zero",
    "(layer) in-consistent input, number of neurons of previous layer, and dimension of neurons of current layer should be equal",
    "only 2 activation function states are implemented for now",
    "(neuron) invalid input, value can't be converted to positive integer",
   "(neuron) integer value wasn't supposed to be zero",
   "(neuron) in-consistent input, dimension of neuron doesn't match the dimension specified in layer description",
   "(neuron) invalid input, value can't be converted to float",
};

void saveMLP(MLP *mlp, const char *Fname) {
  FILE *file;
  file = fopen(Fname, "w");
  if (file == NULL) {
    printf("Error opening file %s\n", Fname);
    exit(1);
  }

  fprintf(file, "%s\n", mlp->name);

  fprintf(file, "%d\n%d\n%d\n", mlp->num_of_outputs, mlp->num_of_inputs,
          mlp->num_of_layers);

  for (int i = 0; i < mlp->num_of_layers; i++) {
    fprintf(file, "%s\n", mlp->layers[i]->name);

    fprintf(file, "%d\n%d\n%d\n", mlp->layers[i]->num_of_neurons,
            mlp->layers[i]->dim_of_neurons, mlp->actfunc[i]);

    for (int j = 0; j < mlp->layers[i]->num_of_neurons; j++) {

      fprintf(file, "%s\n", mlp->layers[i]->neurons[j]->name);

      fprintf(file, "%d\n", mlp->layers[i]->neurons[j]->dimension);

      fprintf(file, "%f\n", mlp->layers[i]->neurons[j]->bias->data);

      for (int k = 0; k < mlp->layers[i]->neurons[j]->dimension; k++) {
        fprintf(file, "%f\n", mlp->layers[i]->neurons[j]->weights[k]->data);
      }
    }
  }
  fclose(file);
}

MLP *loadMLP(const char *Fname) {
  char line[20];
  FILE *file;
  file = fopen(Fname, "r");
  if (file == NULL) {
    printf("Error opening file %s\n", Fname);
    return NULL;
  }
  int ecode = validate(Fname);
  if (ecode) {
    printf("Error parsing file %s : %s\n", Fname, error_messages[ecode]);
    return NULL;
  }
  MLP *mlp = malloc(sizeof(MLP));

  fgets(mlp->name, sizeof(mlp->name), file);
  mlp->name[strcspn(mlp->name, "\n")] = '\0';

  fgets(line, sizeof(line), file);
  mlp->num_of_outputs = atoi(line);

  fgets(line, sizeof(line), file);
  mlp->num_of_inputs = atoi(line);

  fgets(line, sizeof(line), file);
  mlp->num_of_layers = atoi(line);

  mlp->layers = malloc(sizeof(Layer *) * mlp->num_of_layers);
  mlp->actfunc = malloc(sizeof(int) * mlp->num_of_layers);
  for (int i = 0; i < mlp->num_of_layers; i++) {
    mlp->layers[i] = malloc(sizeof(Layer));
    fgets(mlp->layers[i]->name, sizeof(mlp->layers[i]->name), file);
    mlp->layers[i]->name[strcspn(mlp->layers[i]->name, "\n")] = '\0';

    fgets(line, sizeof(line), file);
    mlp->layers[i]->num_of_neurons = atoi(line);
    fgets(line, sizeof(line), file);
    mlp->layers[i]->dim_of_neurons = atoi(line);
    fgets(line, sizeof(line), file);
    mlp->actfunc[i] = atoi(line);

    mlp->layers[i]->neurons =
        malloc(sizeof(Neuron *) * mlp->layers[i]->num_of_neurons);

    for (int j = 0; j < mlp->layers[i]->num_of_neurons; j++) {
      mlp->layers[i]->neurons[j] = malloc(sizeof(Neuron));

      fgets(mlp->layers[i]->neurons[j]->name,
            sizeof(mlp->layers[i]->neurons[j]->name), file);
      mlp->layers[i]
          ->neurons[j]
          ->name[strcspn(mlp->layers[i]->neurons[j]->name, "\n")] = '\0';

      fgets(line, sizeof(line), file);
      mlp->layers[i]->neurons[j]->dimension = atoi(line);

      fgets(line, sizeof(line), file);
      mlp->layers[i]->neurons[j]->bias = createNewValue(atof(line), "b");
      mlp->layers[i]->neurons[j]->bias->_isparameter = 1;

      mlp->layers[i]->neurons[j]->weights =
          malloc(sizeof(Value *) * mlp->layers[i]->neurons[j]->dimension);

      for (int k = 0; k < mlp->layers[i]->neurons[j]->dimension; k++) {
        fgets(line, sizeof(line), file);
        mlp->layers[i]->neurons[j]->weights[k] =
            createNewValue(atof(line), "w");
        snprintf(mlp->layers[i]->neurons[j]->weights[k]->name,
                 sizeof(mlp->layers[i]->neurons[j]->weights[k]->name), "w%d",
                 i);
        mlp->layers[i]->neurons[j]->weights[k]->_isparameter = 1;
      }
    }
  }
  fclose(file);
  return mlp;
}

typedef enum {
  MLP_DESCRIPTION,
  LAYER_DESCRIPTION,
  NEURON_DESCRIPTION
} parsing_state;
typedef struct {
  char *mlp_name, *layer_name, *neuron_name;
  size_t mlp_num_of_inputs, mlp_num_of_outputs, mlp_num_of_layers,
      prev_layer_num_of_neurons, curr_layer_dim_of_neuron,
      curr_layer_num_of_neuron, curr_layer_act, neuron_dim, neuron_count,
      neuron_param_index, neuron_index, layer_index;
  parsing_state state;
} parsing_data;

bool is_positive_int(char *str) {
  // Assuming null terminated string
  size_t start = 0, end = 0;
  while (*(str + end))
    end++;
  // left strip
  while (isspace((unsigned char)*(str + start)))
    start++;
  // right strip
  if (!end)
    return false; // empty string
  end--;
  while (end > start && isspace((unsigned char)*(str + end)))
    end--;
  if (isspace((unsigned char)*(str + end)))
    return false;
  for (size_t i = start; i <= end; i++) {
    if (!isdigit((unsigned char)*(str + i)))
      return false;
  }
  return true;
}

bool is_float(char *str) {
  // Assuming null terminated string
  size_t start = 0, end = 0;
  while (*(str + end))
    end++;
  // left strip
  while (isspace((unsigned char)*(str + start)))
    start++;
  // right strip
  if (!end)
    return false; // empty string
  end--;
  while (end > start && isspace((unsigned char)*(str + end)))
    end--;
  if (isspace((unsigned char)*(str + end)))
    return false;

  bool point_read = false;
  for (size_t i = start; i <= end; i++) {
    if (*(str + i) == '-' && !i)
      continue;
    if (*(str + i) == '.' && !point_read) {
      point_read = true;
      continue;
    }
    if (!isdigit((unsigned char)*(str + i)))
      return false;
  }
  return point_read;
}

int validate(const char *Fname) {
  FILE *file = fopen(Fname, "r");
  if (!file)
    return 1;
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  rewind(file);
  char *buf = malloc(sizeof(char) * size);
  if (!buf) {
    printf("unable to validate file %s ran out of memory\n", Fname);
    exit(EXIT_FAILURE);
  }
  size_t read = fread(buf, 1, size, file);
  fclose(file);
  buf[read] = '\0';
  // main validation starts from here
  parsing_data parsing;
  parsing.state = MLP_DESCRIPTION;
  parsing.mlp_name = NULL;
  parsing.layer_name = NULL;
  parsing.neuron_name = NULL;
  parsing.mlp_num_of_inputs = 0;
  parsing.mlp_num_of_outputs = 0;
  parsing.mlp_num_of_layers = 0;
  parsing.prev_layer_num_of_neurons = 0;
  parsing.curr_layer_dim_of_neuron = 0;
  parsing.curr_layer_num_of_neuron = 0;
  parsing.curr_layer_act = -1;
  parsing.neuron_count = 0;
  parsing.neuron_dim = 0;
  parsing.neuron_param_index = 0;
  parsing.neuron_index = 0;
  parsing.layer_index = 0;
  size_t i=0, line_start, line_end, line_len, line_num = 0;
  while (buf[i]) {
    line_num++;
    line_start = i;
    while (buf[i] && buf[i] != '\n')
      i++;

    line_end = i;
    i++;
    line_len = line_end - line_start;
    char line[line_len + 1];
    for (size_t j = 0; j < line_len; j++)
      line[j] = buf[line_start + j];

    line[line_len] = '\0';
    //printf("(%zu) %s\n",line_len, line);

    switch (parsing.state) {
    case MLP_DESCRIPTION:
      if (!parsing.mlp_name) {
        if (line_len > 5) {
          return 1;
        }
        parsing.mlp_name = line;
      } else if (!parsing.mlp_num_of_outputs) {
        if (!is_positive_int(line))
          return 2;
        if (!atoi(line))
          return 3;
        parsing.mlp_num_of_outputs = atoi(line);
      } else if (!parsing.mlp_num_of_inputs) {
        if (!is_positive_int(line))
          return 2;
        if (!atoi(line))
          return 3;
        parsing.mlp_num_of_inputs = atoi(line);
      } else if (!parsing.mlp_num_of_layers) {
        if (!is_positive_int(line))
          return 2;
        if (!atoi(line))
          return 3;
        parsing.mlp_num_of_layers = atoi(line);
        parsing.prev_layer_num_of_neurons = parsing.mlp_num_of_inputs;
        parsing.state = LAYER_DESCRIPTION;
      }
      break;
    case LAYER_DESCRIPTION:
      if (!parsing.layer_name) {
        if (line_len > 5) {
          return 4;
        }
        parsing.layer_name = line;
      } else if (!parsing.curr_layer_num_of_neuron) {
        if (!is_positive_int(line))
          return 5;
        if (!atoi(line))
          return 6;
        parsing.curr_layer_num_of_neuron = atoi(line);
      } else if (!parsing.curr_layer_dim_of_neuron) {
        if (!is_positive_int(line))
          return 5;
        if (!atoi(line))
          return 6;
        parsing.curr_layer_dim_of_neuron = atoi(line);
        if (parsing.curr_layer_dim_of_neuron !=
            parsing.prev_layer_num_of_neurons){
		printf("%zu %zu\n", parsing.curr_layer_dim_of_neuron, parsing.prev_layer_num_of_neurons);
          return 7;
	}
      } else if (parsing.curr_layer_act < 0) {
        if (!is_positive_int(line))
          return 5;
        if (!atoi(line))
          return 6;
        parsing.curr_layer_act = atoi(line);
        if (parsing.curr_layer_act > 2)
          return 8;
      }
      break;
    case NEURON_DESCRIPTION:
      if (!parsing.neuron_name) {
        if (line_len > 5) {
          return 4;
        }
        parsing.neuron_name = line;
      } else if (!parsing.neuron_dim) {
        if (!is_positive_int(line))
          return 9;
        if (!atoi(line))
          return 10;
        parsing.neuron_dim = atoi(line);
        if(parsing.neuron_dim != parsing.curr_layer_dim_of_neuron)
          return 11;
      } else {
        if (!is_float(line))
          return 12;
        parsing.neuron_param_index++;
        if (parsing.neuron_param_index == parsing.neuron_dim + 1){
          parsing.neuron_count++;
          parsing.neuron_name = NULL;
          parsing.neuron_dim = 0;
          parsing.neuron_param_index = 0;
          if (parsing.curr_layer_num_of_neuron == parsing.neuron_count){
            parsing.prev_layer_num_of_neurons = parsing.neuron_count;
            parsing.layer_name = NULL;
            parsing.curr_layer_num_of_neuron = 0;
            parsing.curr_layer_dim_of_neuron = 0;
            parsing.curr_layer_act = 0;
            parsing.layer_index++;
            if(parsing.mlp_num_of_layers == parsing.layer_index) break;
            parsing.state = LAYER_DESCRIPTION;
          }
        }
        // count dim + 1 valid floats
        // also update neuron count
        // then reset for next neuron / layer
        // based on that
      }
      break;
    }
  }
  return 0;
}
