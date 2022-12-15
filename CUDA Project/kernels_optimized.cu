#include <stdio.h>
#include "fann.h"
#include <string.h>
#include <time.h>

FANN_EXTERNAL fann_type *FANN_API fann_run_parallel(struct fann *ann, fann_type *input, fann_type *weights_device, 
                        struct fann *ann_device);
float fann_train_epoch_irpropm_parallel(struct fann *ann, struct fann_train_data *data);
void fann_backpropagate_MSE_parallel(struct fann *ann, struct fann *ann_device, fann_type *weights_device, fann_type *train_error_device);
void fann_compute_MSE_parallel(struct fann *ann, fann_type *desired_output, struct fann *ann_device, fann_type *train_error_device); 
__global__ void neuron_computation(struct fann_layer *layer_it,struct fann *ann, struct fann_neuron **neuron_pointers,
    bool memcopy_neuron, struct fann_neuron *neurons, fann_type *weights_device, struct fann_neuron *neurons_device,
    int num_blocks, int total_work);
__global__ void kernel_MSE(struct fann *ann, struct fann_neuron *last_layer, fann_type *output_device,
                            fann_type *train_error_device, int pos_prev);
__global__ void neuron_backpropagate(struct fann_neuron *neurons_device,fann_type *weights_device, 
  int pos_prev, fann_type *error_begin, int pos_current_layer);
__global__ void backpropagate_accumulate_last(struct fann_neuron *prev_layer_first_neuron_device, int pos_prev, 
                                              fann_type *error_begin);
void fann_update_slopes_batch_parallel(struct fann *ann, struct fann_layer *layer_begin, struct fann_layer *layer_end, 
                fann_type *train_error_device, fann_type *slopes_device);
__global__ void neuron_update(struct fann_neuron *last_layer_device,int pos_current_layer, 
                  struct fann_neuron *neurons_device, fann_type *slope_begin, fann_type *error_begin,
                  int num_blocks, int total_work);

__global__ void fann_mult_kernel(fann_type* weights_device,struct fann_neuron* neurons, fann_type *to_be_added);

__global__ void array_add(fann_type *arr, int N, fann_type *sum);

// float fann_train_epoch_irpropm(struct fann *ann, struct fann_train_data *data);

// For sequential 
float fann_train_epoch_irpropm(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;

  if (ann->prev_train_slopes == NULL) {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);
  float fann_run_time = 0.0, mse_time = 0.0, backprop_time = 0.0, update_time=0.0;
  for (i = 0; i < data->num_data; i++) {
    clock_t start_fann_run = clock();
    fann_run(ann, data->input[i]);
    clock_t end_fann_runn = clock();
    fann_run_time += (float)(end_fann_runn - start_fann_run) / CLOCKS_PER_SEC;
    clock_t start_MSE = clock();
    fann_compute_MSE(ann, data->output[i]);
    clock_t end_MSE = clock();
    mse_time += (float)(end_MSE-start_MSE)/CLOCKS_PER_SEC;
    clock_t start_backprop = clock();
    fann_backpropagate_MSE(ann);
    clock_t end_backprop = clock();
    backprop_time += (float)(end_backprop-start_backprop)/CLOCKS_PER_SEC;
    clock_t start_update = clock();
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
    clock_t end_update = clock();
    update_time += (float)(end_update-start_update)/CLOCKS_PER_SEC;
  }

  fann_update_weights_irpropm(ann, 0, ann->total_connections);
  printf("fann_run: %f \n" ,fann_run_time);
  printf("fann_compute_MSE: %f \n", mse_time);
  printf("fann_backpropagate_MSE: %f \n", backprop_time);
  printf("fann_update_slopes_batch: %f \n", update_time);

  return fann_get_MSE(ann);
}


int main(int argc, char* argv[])
{ 
  unsigned int num_layers=3, num_neurons_hidden=32;
  float desired_error=0.001;
  // printf("%d args (1:%s) \n", argc, argv[1]);
  char mushroom[20] = "mushroom";
  char robot[20] = "robot";
  char kin32fm[20] = "kin32fm";
  if (strcmp(argv[1],mushroom) == 0){
    printf("Using mushroom dataset\n");
    num_layers = 3;
    num_neurons_hidden = 32;
    desired_error = 0.0001;
  }
  else if (strcmp(argv[1], robot) == 0){
    num_layers = 3;
    num_neurons_hidden = 64;
    desired_error = 0.001;
  }
  else if (strcmp(argv[1],kin32fm) == 0){
    num_layers = 3;
    num_neurons_hidden = 64;
    desired_error = 0.001;
  }
	const unsigned int max_epochs = 50;
	const unsigned int epochs_between_reports = 1;
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;

	unsigned int i = 0;
	printf("Creating network.\n");
  char path[20] = "../datasets/", tr[10] = ".train";
	train_data = fann_read_train_from_file(strcat(strcat(path,argv[1]), tr));

	ann = fann_create_standard(num_layers,
					  // train_data->num_input, num_neurons_hidden*2, num_neurons_hidden,train_data->num_output);
					  train_data->num_input,num_neurons_hidden,train_data->num_output);

  ann->train_errors = (fann_type *)calloc(ann->total_neurons, sizeof(fann_type));
  if (ann->train_errors == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return 1;
  }

  /* if no room allocated for the slope variabels, allocate it now */

  ann->train_slopes = (fann_type *)calloc(ann->total_connections_allocated, sizeof(fann_type));
  if (ann->train_slopes == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }
  

	printf("Training network.\n");

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID);

	/*fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL); */

  clock_t start_train = clock();
	fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);
  clock_t end_train = clock();
  printf("fann_train_on_data: %f \n", (float)(end_train-start_train)/CLOCKS_PER_SEC);

	printf("Testing network.\n");

  char path2[20] = "../datasets/", te2[10] = ".test";
	test_data = fann_read_train_from_file(strcat(strcat(path2,argv[1]),te2));

	fann_reset_MSE(ann);
	for(i = 0; i < fann_length_train_data(test_data); i++)
	{
		fann_test(ann, test_data->input[i], test_data->output[i]);
	}
	
	printf("MSE error on test data: %f\n", fann_get_MSE(ann));

	printf("Saving network.\n");

	fann_save(ann, "mushroom_float.net");

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);

	return 0;
}

FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
                                               unsigned int max_epochs,
                                               unsigned int epochs_between_reports,
                                               float desired_error) {
  float error;
  unsigned int i;
  int desired_error_reached;

#ifdef DEBUG
  printf("Training with %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
#endif

  if (epochs_between_reports && ann->callback == NULL) {
    printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);
  }

  for (i = 1; i <= max_epochs; i++) {
    /*
     * train
     */
    //Sequential 
    // clock_t start_epoch = clock();
    // error = fann_train_epoch_irpropm(ann,data);
    // clock_t end_epoch = clock();
    // printf("fann_train_epoch_irpropm: %f \n",(float)(end_epoch-start_epoch)/CLOCKS_PER_SEC);
    //Parallel
    clock_t start_epoch = clock();
    error = fann_train_epoch_irpropm_parallel(ann, data);
    clock_t end_epoch = clock();
    printf("fann_train_epoch_irpropm_parallel: %f \n",(float)(end_epoch-start_epoch)/CLOCKS_PER_SEC);
    desired_error_reached = fann_desired_error_reached(ann, desired_error);
    /*
     * print current output
     */
    if (epochs_between_reports && (i % epochs_between_reports == 0 || i == max_epochs || i == 1 ||
                                   desired_error_reached == 0)) {
      if (ann->callback == NULL) {
        printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error, ann->num_bit_fail);
      } else if (((*ann->callback)(ann, data, max_epochs, epochs_between_reports, desired_error,
                                   i)) == -1) {
        /*
         * you can break the training by returning -1
         */
        break;
      }
    }

    if (desired_error_reached == 0) break;
  }
}

float fann_train_epoch_irpropm_parallel(struct fann *ann, struct fann_train_data *data) {
  // printf("fann_train_epoch_irpropm modified running");
  unsigned int i;

  if (ann->prev_train_slopes == NULL) {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  //Parallel Implementation 
  fann_type *weights_device;
  cudaMalloc((void **)&weights_device, sizeof(fann_type)*ann->total_connections);
  cudaMemcpy(weights_device, ann->weights, sizeof(fann_type)*ann->total_connections, cudaMemcpyHostToDevice);
  
  struct fann *ann_device;
  cudaMalloc((void **)&ann_device, sizeof(struct fann ));
  cudaMemcpy(ann_device, ann , sizeof(struct fann), cudaMemcpyHostToDevice );
  
  fann_type *train_error_device;
  cudaMalloc((void **)&train_error_device, sizeof(fann_type)*ann->total_neurons);

  fann_type *output_device;
  cudaMalloc((void **)&output_device, sizeof(fann_type)*data->num_data*data->num_output);

  fann_type *slopes_device;
  cudaMalloc((void **)&slopes_device, sizeof(fann_type)*ann->total_connections_allocated);
  float fann_run_time = 0.0, mse_time = 0.0, backprop_time = 0.0, update_time=0.0;
  for (i = 0; i < data->num_data; i++) {
    clock_t start_fann_run = clock();
    fann_run_parallel(ann, data->input[i], weights_device, ann_device);
    clock_t end_fann_runn = clock();
    fann_run_time += (float)(end_fann_runn - start_fann_run) / CLOCKS_PER_SEC;
    clock_t start_MSE = clock();
    cudaMemcpy(&output_device[i],data->output[i],sizeof(fann_type)*data->num_output,cudaMemcpyHostToDevice);
    fann_compute_MSE_parallel(ann, &output_device[i], ann_device,train_error_device);
    clock_t end_MSE = clock();
    mse_time += (float)(end_MSE-start_MSE)/CLOCKS_PER_SEC;
    clock_t start_backprop = clock();
    fann_backpropagate_MSE_parallel(ann,ann_device,weights_device,train_error_device);
    clock_t end_backprop = clock();
    backprop_time += (float)(end_backprop-start_backprop)/CLOCKS_PER_SEC;
    clock_t start_update = clock();
    fann_update_slopes_batch_parallel(ann, ann->first_layer + 1, ann->last_layer - 1,train_error_device,slopes_device);
    clock_t end_update = clock();
    update_time += (float)(end_update-start_update)/CLOCKS_PER_SEC;
  }
  cudaMemcpy(ann->weights, weights_device, sizeof(fann_type)*ann->total_connections, cudaMemcpyDeviceToHost);
  cudaMemcpy(ann->train_slopes, slopes_device, sizeof(fann_type)*ann->total_connections_allocated, cudaMemcpyDeviceToHost);
  cudaFree(weights_device);
  cudaFree(output_device);
  cudaFree(ann_device);
  cudaFree(train_error_device);
  cudaFree(slopes_device);
  fann_update_weights_irpropm(ann, 0, ann->total_connections);
    printf("fann_run time: %f \n" ,fann_run_time);
    // printf("fann_compute_MSE_parallel: %f \n", mse_time);
    // printf("fann_backpropagate_MSE_parallel: %f \n", backprop_time);
    printf("fann_update_slopes_batch_parallel: %f \n", update_time);

  return fann_get_MSE(ann);
}

FANN_EXTERNAL fann_type *FANN_API fann_run_parallel(struct fann *ann, fann_type *input, fann_type *weights_device, 
                        struct fann *ann_device) {
// printf("fann_run started");
  struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers, *neurons_device, *prev_layer_first_neuron_device;
  unsigned int i, num_input, num_output;
  fann_type *output;
  struct fann_layer *layer_it, *last_layer, *layer_it_device;

//   unsigned int activation_function;
//   fann_type steepness;

  /* store some variabels local for fast access */
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;


#ifdef FIXEDFANN
  int multiplier = ann->multiplier;
  unsigned int decimal_point = ann->decimal_point;

  /* values used for the stepwise linear sigmoid function */
  fann_type r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
  fann_type v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;

  fann_type last_steepness = 0;
  unsigned int last_activation_function = 0;
#else
  fann_type max_sum = 0;
#endif

  /* first set the input */
  num_input = ann->num_input;
  for (i = 0; i != num_input; i++) {
#ifdef FIXEDFANN
    if (fann_abs(input[i]) > multiplier) {
      printf(
          "Warning input number %d is out of range -%d - %d with value %d, integer overflow may "
          "occur.\n",
          i, multiplier, multiplier, input[i]);
    }
#endif
    first_neuron[i].value = input[i];
  }
  /* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
  (ann->first_layer->last_neuron - 1)->value = multiplier;
#else
  (ann->first_layer->last_neuron - 1)->value = 1;
#endif
  // printf("Before layer_it loop");
  last_layer = ann->last_layer;
  
  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
    int num_layer_neurons = layer_it->last_neuron-layer_it->first_neuron -1;
    int num_threads = num_layer_neurons;
    // if ((num_layer_neurons/8)<1){
    //   num_threads = 1;
    // }
    // else{
    //   num_threads = (int)num_layer_neurons/8;
    // }
    // printf("threads %d \n",num_threads);
    // printf("num layer neurons %d \n",num_layer_neurons);
    // This is for the default setting (network type is not FANN_NETTYPE_SHORTCUT, it is FANN_NETTYPE_LAYER)
    neurons = (layer_it - 1)->first_neuron;
    
    // printf("starting 0");
    cudaMalloc((void **)&prev_layer_first_neuron_device, ((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron -1 )*sizeof(struct fann_neuron));
    // printf("starting 0-memcpy");
    cudaMemcpy(prev_layer_first_neuron_device, neurons, ((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron -1 )*sizeof(struct fann_neuron), cudaMemcpyHostToDevice);
    cudaError_t err0 = cudaGetLastError();
    if(err0 != cudaSuccess)
      printf("Error 0 %s\n",cudaGetErrorString(err0));
    // printf("layer_it loop started");
    // cudaMalloc((void **)&layer_it_device, sizeof(struct fann_layer ));
    
    cudaError_t err1 = cudaGetLastError();
    if(err1 != cudaSuccess)
      printf("Error 1 %s\n",cudaGetErrorString(err1));
    cudaMalloc((void **)&neurons_device, sizeof(struct fann_neuron)*num_layer_neurons);
    cudaError_t err2 = cudaGetLastError();
    if(err2 != cudaSuccess)
      printf("Error 2 %s\n",cudaGetErrorString(err2));
    
    first_neuron = layer_it->first_neuron;
    last_neuron = layer_it->last_neuron;
    int num_blocks = 1;

    // Memory copy 
    bool memcopy_neuron = true; //To access memory neuron wise, set to false to copy more memory at once
    // memcpy and cudamalloc (layer_it)
    // cudaMemcpy(layer_it_device, layer_it , sizeof(struct fann_layer ), cudaMemcpyHostToDevice );
    
    cudaError_t err3 = cudaGetLastError();
    if(err3 != cudaSuccess)
      printf("Error 3 %s\n",cudaGetErrorString(err3));
    cudaMemcpy(neurons_device, layer_it->first_neuron,sizeof(struct fann_neuron)*num_layer_neurons,cudaMemcpyHostToDevice);
    cudaError_t err4 = cudaGetLastError();
    if(err4 != cudaSuccess)
      printf("Error 4 %s\n",cudaGetErrorString(err4));
    
    neuron_computation<<<num_blocks,num_threads>>>(layer_it_device,ann_device,neuron_pointers,
            memcopy_neuron,prev_layer_first_neuron_device,weights_device, neurons_device, num_blocks, num_layer_neurons);
    cudaMemcpy(layer_it->first_neuron, neurons_device,sizeof(struct fann_neuron)*num_layer_neurons,cudaMemcpyDeviceToHost);
    // cudaMemcpy(ann->weights, weights_device, sizeof(ann->weights), cudaMemcpyDeviceToHost);
    // cudaMemcpy(neurons, prev_layer_first_neuron_device, sizeof(struct fann_neuron), cudaMemcpyDeviceToHost);
    // cudaMemcpy(layer_it, layer_it_device, sizeof(struct fann_layer ), cudaMemcpyDeviceToHost );
    // cudaMemcpy(ann, ann_device, sizeof(struct fann), cudaMemcpyDeviceToHost );
    // cudaFree(layer_it_device);
    cudaFree(prev_layer_first_neuron_device);    
    cudaFree(neurons_device);    
  }
  

  /* set the output */
  output = ann->output;
  num_output = ann->num_output;
  neurons = (ann->last_layer - 1)->first_neuron;
  for (i = 0; i != num_output; i++) {
    output[i] = neurons[i].value;
  }
  return ann->output;
}

__global__ void neuron_computation(struct fann_layer *layer_it,struct fann *ann, struct fann_neuron **neuron_pointers,
    bool memcopy_neuron, struct fann_neuron *neurons, fann_type *weights_device, struct fann_neuron *neurons_device,
    int num_blocks, int total_work){
    // for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
    
  int thread_work = total_work/(num_blocks*blockDim.x);
  for (int w=0; w<thread_work; ++w)
  {
    // Method 1: No memory coalescing
    // int base = w+ thread_work*(blockIdx.x*blockDim.x +threadIdx.x);
    // Method 2: With memory coalescing 
    int base =  w*thread_work+ blockIdx.x*blockDim.x +threadIdx.x;
    // struct fann_neuron *neuron_it =  + neurons_device;
    
    struct fann_neuron *neuron_it = base + neurons_device;
    unsigned int i;
    fann_type max_sum = 0;


      if (neuron_it->first_con == neuron_it->last_con) {
        /* bias neurons */
    #ifdef FIXEDFANN
            neuron_it->value = multiplier;
    #else
            neuron_it->value = 1;
    #endif
            return;
        }

      unsigned int activation_function = neuron_it->activation_function;
      fann_type steepness = neuron_it->activation_steepness;

      fann_type neuron_sum = 0;
      unsigned int num_connections = neuron_it->last_con - neuron_it->first_con;
      fann_type *weights;
      weights = weights_device + neuron_it->first_con;
      if (ann->connection_rate >= 1) {
        if (ann->network_type == FANN_NETTYPE_SHORTCUT) {
          neurons = ann->first_layer->first_neuron;
          printf("THIS SHOULD NOT BE PRINTED");
        } 
        // else {
        //   neurons = (layer_it - 1)->first_neuron;
        // }

        /* unrolled loop start */

        // Optimized fann_mult
        // fann_type *to_be_added, *sum;
        // cudaMalloc((void **)&to_be_added, (num_connections-1)*sizeof(fann_type));
        // cudaMalloc((void **)&sum, sizeof(fann_type));

        // fann_mult_kernel<<<1,num_connections-1>>>(weights,neurons,to_be_added);
        // array_add<<<1,1>>>(to_be_added, num_connections-1, sum);

        // cudaFree(to_be_added);  
        // neuron_sum = *sum;
        // cudaFree(sum);  
        // original loop for fann_mult
        i = num_connections & 3; /* same as modulo 4 */
        switch (i) {
          case 3:
            neuron_sum += fann_mult(weights[2], neurons[2].value);
          case 2:
            neuron_sum += fann_mult(weights[1], neurons[1].value);
          case 1:
            neuron_sum += fann_mult(weights[0], neurons[0].value);
          case 0:
            break;
        }

        for (; i != num_connections-4; i += 4) {
          neuron_sum += fann_mult(weights[i], neurons[i].value) +
                        fann_mult(weights[i + 1], neurons[i + 1].value) +
                        fann_mult(weights[i + 2], neurons[i + 2].value) +
                        fann_mult(weights[i + 3], neurons[i + 3].value);
        }
        /* unrolled loop end */

        
        // for(i = 0;i != num_connections -1; i++){
        //   // printf("%f += %f*%f, ", neuron_sum, weights[i], neurons[i].value);
        //   neuron_sum += fann_mult(weights[i], neurons[i].value);
        // }
         
      } else {
        // neuron_pointers = ann->connections + *device_neuron_it->first_con;
        neuron_pointers = ann->connections + neuron_it->first_con;

        i = num_connections & 3; /* same as modulo 4 */
        switch (i) {
          case 3:
            neuron_sum += fann_mult(weights[2], neuron_pointers[2]->value);
          case 2:
            neuron_sum += fann_mult(weights[1], neuron_pointers[1]->value);
          case 1:
            neuron_sum += fann_mult(weights[0], neuron_pointers[0]->value);
          case 0:
            break;
        }

        for (; i != num_connections; i += 4) {
          neuron_sum += fann_mult(weights[i], neuron_pointers[i]->value) +
                        fann_mult(weights[i + 1], neuron_pointers[i + 1]->value) +
                        fann_mult(weights[i + 2], neuron_pointers[i + 2]->value) +
                        fann_mult(weights[i + 3], neuron_pointers[i + 3]->value);
        }
        
      }

#ifdef FIXEDFANN
      neuron_it->sum = fann_mult(steepness, neuron_sum);

      if (activation_function != last_activation_function || steepness != last_steepness) {
        switch (activation_function) {
          case FANN_SIGMOID:
          case FANN_SIGMOID_STEPWISE:
            r1 = ann->sigmoid_results[0];
            r2 = ann->sigmoid_results[1];
            r3 = ann->sigmoid_results[2];
            r4 = ann->sigmoid_results[3];
            r5 = ann->sigmoid_results[4];
            r6 = ann->sigmoid_results[5];
            v1 = ann->sigmoid_values[0] / steepness;
            v2 = ann->sigmoid_values[1] / steepness;
            v3 = ann->sigmoid_values[2] / steepness;
            v4 = ann->sigmoid_values[3] / steepness;
            v5 = ann->sigmoid_values[4] / steepness;
            v6 = ann->sigmoid_values[5] / steepness;
            break;
          case FANN_SIGMOID_SYMMETRIC:
          case FANN_SIGMOID_SYMMETRIC_STEPWISE:
            r1 = ann->sigmoid_symmetric_results[0];
            r2 = ann->sigmoid_symmetric_results[1];
            r3 = ann->sigmoid_symmetric_results[2];
            r4 = ann->sigmoid_symmetric_results[3];
            r5 = ann->sigmoid_symmetric_results[4];
            r6 = ann->sigmoid_symmetric_results[5];
            v1 = ann->sigmoid_symmetric_values[0] / steepness;
            v2 = ann->sigmoid_symmetric_values[1] / steepness;
            v3 = ann->sigmoid_symmetric_values[2] / steepness;
            v4 = ann->sigmoid_symmetric_values[3] / steepness;
            v5 = ann->sigmoid_symmetric_values[4] / steepness;
            v6 = ann->sigmoid_symmetric_values[5] / steepness;
            break;
          case FANN_THRESHOLD:
            break;
        }
      }

      switch (activation_function) {
        case FANN_SIGMOID:
        case FANN_SIGMOID_STEPWISE:
          neuron_it->value = (fann_type)fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5,
                                                      r6, 0, multiplier, neuron_sum);
          break;
        case FANN_SIGMOID_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
          neuron_it->value = (fann_type)fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5,
                                                      r6, -multiplier, multiplier, neuron_sum);
          break;
        case FANN_THRESHOLD:
          neuron_it->value = (fann_type)((neuron_sum < 0) ? 0 : multiplier);
          break;
        case FANN_THRESHOLD_SYMMETRIC:
          neuron_it->value = (fann_type)((neuron_sum < 0) ? -multiplier : multiplier);
          break;
        case FANN_LINEAR:
          neuron_it->value = neuron_sum;
          break;
        case FANN_LINEAR_PIECE:
          neuron_it->value = (fann_type)(
              (neuron_sum < 0) ? 0 : (neuron_sum > multiplier) ? multiplier : neuron_sum);
          break;
        case FANN_LINEAR_PIECE_SYMMETRIC:
          neuron_it->value = (fann_type)((neuron_sum < -multiplier)
                                             ? -multiplier
                                             : (neuron_sum > multiplier) ? multiplier : neuron_sum);
          break;
        case FANN_ELLIOT:
        case FANN_ELLIOT_SYMMETRIC:
        case FANN_GAUSSIAN:
        case FANN_GAUSSIAN_SYMMETRIC:
        case FANN_GAUSSIAN_STEPWISE:
        case FANN_SIN_SYMMETRIC:
        case FANN_COS_SYMMETRIC:
          fann_error((struct fann_error *)ann, FANN_E_CANT_USE_ACTIVATION);
          break;
      }
      last_steepness = steepness;
      last_activation_function = activation_function;
#else
      neuron_sum = fann_mult(steepness, neuron_sum);

      max_sum = 150 / steepness;
      if (neuron_sum > max_sum)
        neuron_sum = max_sum;
      else if (neuron_sum < -max_sum)
        neuron_sum = -max_sum;

      neuron_it->sum = neuron_sum;

      fann_activation_switch(activation_function, neuron_sum, neuron_it->value);
#endif
  }
}

__global__ void fann_mult_kernel(fann_type* weights_device,struct fann_neuron* neurons, fann_type *to_be_added){
  fann_type* weights = threadIdx.x +blockIdx.x*blockDim.x + weights_device;
  struct fann_neuron* n = threadIdx.x +blockIdx.x*blockDim.x + neurons;
  // printf("fann_mult kernel %f ", neurons_device->sum);
  // printf("after %f\n",(neurons_device+base)->sum);
  // printf("ne %f\n",neurons->value);
  // printf("n %f\n",n->value);
  // printf("w %f\n",*weights);
   *(to_be_added + threadIdx.x +blockIdx.x*blockDim.x) = fann_mult(*weights, n->value);
}

__global__ void array_add(fann_type *arr, int N, fann_type *sum) {
  for (int i = 0; i < N-1; ++i)
    *sum += arr[i];
}

void fann_compute_MSE_parallel(struct fann *ann, fann_type *desired_output, struct fann *ann_device, fann_type *train_error_device) {
  int num_blocks =1;
  fann_type neuron_value, neuron_diff, *error_it = 0, *error_begin = 0;
  struct fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
  // const struct fann_neuron *last_layer_end = last_layer_begin + ann->num_output;
  struct fann_neuron *last_layer_device;

  int num_last_layer_neurons = (ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron;
  cudaMalloc((void **)&last_layer_device, sizeof(struct fann_neuron)*num_last_layer_neurons);
  cudaMemcpy(last_layer_device,last_layer_begin,sizeof(struct fann_neuron)*num_last_layer_neurons,cudaMemcpyHostToDevice);


  /* clear the error variabels */
  cudaMemset(train_error_device, 0, (ann->total_neurons) * sizeof(fann_type));
  error_begin = ann->train_errors;
  int pos_prev = 0;
  struct fann_layer *temp_layer_it;
  for (temp_layer_it = ann->first_layer; temp_layer_it != ann->last_layer-1; temp_layer_it++) {
    pos_prev += temp_layer_it->last_neuron - temp_layer_it->first_neuron;
  }

#ifdef DEBUGTRAIN
  printf("\ncalculate errors\n");
#endif
  /* calculate the error and place it in the output layer */
  // error_it = error_begin + (last_layer_begin - first_neuron);

  kernel_MSE<<<num_blocks,num_last_layer_neurons>>>(ann_device,last_layer_device, desired_output, train_error_device, pos_prev);
  cudaMemcpy(&ann->MSE_value,&ann_device->MSE_value,sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(&ann->num_bit_fail,&ann_device->num_bit_fail,sizeof(unsigned int),cudaMemcpyDeviceToHost);
  cudaMemcpy(&ann->num_MSE,&ann_device->num_MSE,sizeof(unsigned int),cudaMemcpyDeviceToHost);
  cudaMemcpy(ann->train_errors,train_error_device,sizeof(fann_type)*ann->total_neurons,cudaMemcpyDeviceToHost);
  cudaFree(last_layer_device);
}

__global__ void kernel_MSE(struct fann *ann, struct fann_neuron *last_layer, fann_type *output_device,
                            fann_type *train_error_device, int pos_prev){
  struct fann_neuron *last_layer_begin = threadIdx.x +blockIdx.x*blockDim.x + last_layer;
  // struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  fann_type *error_it = threadIdx.x +blockIdx.x*blockDim.x + train_error_device + pos_prev;
  fann_type *desired_output = threadIdx.x +blockIdx.x*blockDim.x +output_device;
  fann_type neuron_value = last_layer_begin->value;
  fann_type neuron_diff = *desired_output - neuron_value;

  atomicAdd(&ann->MSE_value,neuron_diff*neuron_diff);
  if (fann_abs(neuron_diff) >= ann->bit_fail_limit) {
    atomicAdd(&ann->num_bit_fail,1);
  }

  if (ann->train_error_function) { /* TODO make switch when more functions */
    if (neuron_diff < -.9999999)
      neuron_diff = -17.0;
    else if (neuron_diff > .9999999)
      neuron_diff = 17.0;
    else
      neuron_diff = (fann_type)log((1.0 + neuron_diff) / (1.0 - neuron_diff));
  }

  // *error_it = fann_activation_derived(last_layer_begin->activation_function,
  //                                     last_layer_begin->activation_steepness, neuron_value,
  //                                     last_layer_begin->sum) * neuron_diff;
  fann_type value = (((neuron_value) < (0.01f)) ? (0.01f) : (((neuron_value) > (0.99f)) ? (0.99f) : (neuron_value)));
  *error_it = neuron_diff*(2.0f * last_layer_begin->activation_steepness * value * (1.0f - value));
  atomicAdd(&ann->num_MSE,1);
}

void fann_backpropagate_MSE_parallel(struct fann *ann, struct fann *ann_device, fann_type *weights_device, fann_type *train_error_device) {
  int num_blocks = 1;
  fann_type tmp_error;
  unsigned int i;
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it, *last_neuron, *neurons_device, *prev_layer_first_neuron_device;
  struct fann_neuron **connections;

  fann_type *error_begin = ann->train_errors;
  fann_type *error_prev_layer;
  fann_type *weights;
  const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  const struct fann_layer *second_layer = ann->first_layer + 1;
  struct fann_layer *last_layer = ann->last_layer;

  /* go through all the layers, from last to first.
   * And propagate the error backwards */
  for (layer_it = last_layer - 1; layer_it > second_layer; --layer_it) {
    last_neuron = layer_it->last_neuron;
    int num_layer_neurons = layer_it->last_neuron-layer_it->first_neuron;
    int num_prev_layer_neurons = (layer_it-1)->last_neuron-(layer_it-1)->first_neuron;
    // printf("hello1 with %d\n", num_layer_neurons);

    cudaMalloc((void **)&neurons_device, sizeof(struct fann_neuron)*num_layer_neurons);
    cudaMemcpy(neurons_device, layer_it->first_neuron,sizeof(struct fann_neuron)*num_layer_neurons,cudaMemcpyHostToDevice);
    cudaMalloc((void **)&prev_layer_first_neuron_device, ((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron)*sizeof(struct fann_neuron));
    // printf("starting 0-memcpy");
    cudaMemcpy(prev_layer_first_neuron_device, (layer_it-1)->first_neuron, ((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron)*sizeof(struct fann_neuron), cudaMemcpyHostToDevice);

    /* for each connection in this layer, propagate the error backwards */
    int pos_current_layer = 0;
    if (ann->connection_rate >= 1) {
      if (ann->network_type == FANN_NETTYPE_LAYER) {
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
      } else {
        // Not used in the default setting 
        error_prev_layer = error_begin;
      }
      fann_type *error_prev_layer_device;
      // cudaMalloc((void **)&error_prev_layer_device, sizeof(struct fann_neuron)*num_prev_layer_neurons);
      // cudaMemcpy(error_prev_layer_device, error_prev_layer,sizeof(struct fann_neuron)*num_prev_layer_neurons,cudaMemcpyHostToDevice);
      struct fann_layer *temp_layer_it;
      for (temp_layer_it = ann->first_layer; temp_layer_it != layer_it; temp_layer_it++) {
        pos_current_layer += temp_layer_it->last_neuron - temp_layer_it->first_neuron;
      }
      // printf("hello2 with %d\n", num_layer_neurons);
      neuron_backpropagate<<<num_blocks,num_layer_neurons>>>(neurons_device,weights_device,pos_current_layer-num_prev_layer_neurons,
                                                            train_error_device,pos_current_layer);
    } else {
      // Not used in default setting 
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron];
        weights = ann->weights + neuron_it->first_con;
        connections = ann->connections + neuron_it->first_con;
        for (i = neuron_it->last_con - neuron_it->first_con; i--;) {
          error_begin[connections[i] - first_neuron] += tmp_error * weights[i];
        }
      }
    }
    // cudaMemcpy(error_prev_layer,error_prev_layer_device,sizeof(struct fann_neuron)*num_prev_layer_neurons, cudaMemcpyDeviceToHost);
    /* then calculate the actual errors in the previous layer */
    // error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
    last_neuron = (layer_it - 1)->last_neuron;
    

    // for (neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++) {
    //   *error_prev_layer *=
    //       fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
    //                               neuron_it->value, neuron_it->sum);
    //   error_prev_layer++;
    // }
    backpropagate_accumulate_last<<<num_blocks,((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron)>>>
    (prev_layer_first_neuron_device, pos_current_layer-num_prev_layer_neurons, train_error_device);
    cudaMemcpy(ann->train_errors,train_error_device,sizeof(fann_type)*ann->total_neurons,cudaMemcpyDeviceToHost);
    // cudaFree(error_prev_layer_device);
    cudaFree(neurons_device);
    cudaFree(prev_layer_first_neuron_device);
  }
  // printf("loop ends\n");
}

__global__ void neuron_backpropagate(struct fann_neuron *neurons_device,fann_type *weights_device, 
  int pos_prev, fann_type *error_begin, int pos_current_layer){
  unsigned int i;
  struct fann_neuron *neuron_it = threadIdx.x +blockIdx.x*blockDim.x + neurons_device;
  fann_type tmp_error = *(error_begin + threadIdx.x +blockIdx.x*blockDim.x + pos_current_layer);
  fann_type *error_prev_layer = error_begin + pos_prev;
  fann_type *weights = weights_device + neuron_it->first_con;
  for (i = neuron_it->last_con - neuron_it->first_con; i--;) {
    atomicAdd(&error_prev_layer[i], tmp_error * weights[i]);
  }
}

__global__ void backpropagate_accumulate_last(struct fann_neuron *prev_layer_first_neuron_device, int pos_prev, 
                                              fann_type *error_begin){
  struct fann_neuron *neuron_it = threadIdx.x +blockIdx.x*blockDim.x + prev_layer_first_neuron_device;
  fann_type *error_prev_layer = threadIdx.x +blockIdx.x*blockDim.x + error_begin + pos_prev;

  // *error_prev_layer *=
  //         fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
  //                                 neuron_it->value, neuron_it->sum);
  fann_type value = (((neuron_it->value) < (0.01f)) ? (0.01f) : (((neuron_it->value) > (0.99f)) ? (0.99f) : (neuron_it->value)));
  *error_prev_layer *= (2.0f * neuron_it->activation_steepness * value * (1.0f - value));
}

void fann_update_slopes_batch_parallel(struct fann *ann, struct fann_layer *layer_begin, struct fann_layer *layer_end, 
                fann_type *train_error_device, fann_type *slopes_device) {
  struct fann_neuron *neuron_it, *last_neuron, *prev_neurons, **connections;
  int num_blocks=1;
  fann_type tmp_error;
  unsigned int i, num_connections;

  /* store some variabels local for fast access */
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  fann_type *error_begin = ann->train_errors;
  fann_type *slope_begin, *neuron_slope;

  if (layer_begin == NULL) {
    layer_begin = ann->first_layer + 1;
  }

  if (layer_end == NULL) {
    layer_end = ann->last_layer - 1;
  }

  slope_begin = ann->train_slopes;

#ifdef DEBUGTRAIN
  printf("\nupdate slopes\n");
#endif

  prev_neurons = first_neuron;

  for (; layer_begin <= layer_end; layer_begin++) {
#ifdef DEBUGTRAIN
    printf("layer[%d]\n", layer_begin - ann->first_layer);
#endif
    int num_layer_neurons = layer_begin->last_neuron-layer_begin->first_neuron;
    int num_threads = num_layer_neurons;
    int num_prev_layer_neurons = (layer_begin-1)->last_neuron-(layer_begin-1)->first_neuron;
    last_neuron = layer_begin->last_neuron;
    struct fann_neuron *neurons_device;
    cudaMalloc((void **)&neurons_device, sizeof(struct fann_neuron)*num_layer_neurons);
    cudaMemcpy(neurons_device, layer_begin->first_neuron,sizeof(struct fann_neuron)*num_layer_neurons,cudaMemcpyHostToDevice);
    struct fann_layer *temp_layer_it;
    int pos_current_layer = 0;
    for (temp_layer_it = ann->first_layer; temp_layer_it != layer_begin; temp_layer_it++) {
      pos_current_layer += temp_layer_it->last_neuron - temp_layer_it->first_neuron;
    }
    struct fann_neuron *last_layer_device;
    // int num_last_layer_neurons = 
    cudaMalloc((void **)&last_layer_device, sizeof(struct fann_neuron)*num_prev_layer_neurons);
    cudaMemcpy(last_layer_device,(layer_begin - 1)->first_neuron,sizeof(struct fann_neuron)*num_prev_layer_neurons,cudaMemcpyHostToDevice);

    if (ann->connection_rate >= 1) {
      if (ann->network_type == FANN_NETTYPE_LAYER) {
        prev_neurons = (layer_begin - 1)->first_neuron;
      }
      neuron_update<<<num_blocks, num_threads>>>(last_layer_device, pos_current_layer, neurons_device,
              slopes_device,train_error_device, num_blocks, num_layer_neurons);
      // for (neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++) {
      //   tmp_error = error_begin[neuron_it - first_neuron];
      //   neuron_slope = slope_begin + neuron_it->first_con;
      //   num_connections = neuron_it->last_con - neuron_it->first_con;
      //   for (i = 0; i != num_connections; i++) {
      //     neuron_slope[i] += tmp_error * prev_neurons[i].value;
      //   }
      // }
    } else {
      // not being used in default setting hence not parallelized 
      for (neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron];
        neuron_slope = slope_begin + neuron_it->first_con;
        num_connections = neuron_it->last_con - neuron_it->first_con;
        connections = ann->connections + neuron_it->first_con;
        for (i = 0; i != num_connections; i++) {
          neuron_slope[i] += tmp_error * connections[i]->value;
        }
      }
    }
    cudaFree(neurons_device);
    cudaFree(last_layer_device);
  }

}

__global__ void neuron_update(struct fann_neuron *last_layer_device,int pos_current_layer, 
                  struct fann_neuron *neurons_device, fann_type *slope_begin, fann_type *error_begin,
                  int num_blocks, int total_work){
  int thread_work = total_work/(num_blocks*blockDim.x);
  for (int w=0; w<thread_work; ++w){
    // Method 1: Without memory coalescing 
    // struct fann_neuron *neuron_it = w + thread_work*(threadIdx.x +blockIdx.x*blockDim.x) + neurons_device;
    // fann_type tmp_error = error_begin[w + thread_work*(threadIdx.x +blockIdx.x*blockDim.x) +pos_current_layer ];
    // Method 2: with memory coalescing
    struct fann_neuron *neuron_it = w*thread_work + threadIdx.x +blockIdx.x*blockDim.x + neurons_device;
    fann_type tmp_error = error_begin[w*thread_work + threadIdx.x +blockIdx.x*blockDim.x +pos_current_layer ];
    fann_type *neuron_slope;
    neuron_slope = slope_begin + neuron_it->first_con;
    unsigned int num_connections = neuron_it->last_con - neuron_it->first_con;
    unsigned int i;
    for (i = 0; i != num_connections; i++) {
      atomicAdd(&neuron_slope[i], tmp_error * last_layer_device[i].value);
    }
  }
}