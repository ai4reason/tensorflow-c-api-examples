#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main(int argc, char** argv) 
{
   printf("Hello from TensorFlow C library version %s\n", TF_Version());
   if (argc == 1)
   {
      printf("usage: %s <model_dir>\n", argv[0]);
      return -1;
   }
  
   TF_Graph* graph = TF_NewGraph();
   TF_SessionOptions* options = TF_NewSessionOptions();
   TF_Status* status = TF_NewStatus();

   TF_Buffer* run = TF_NewBuffer();
   TF_Buffer* meta = TF_NewBuffer();

   // load model with default tags
   const char* tags[1] = {"serve"};

   TF_Session* tfs = TF_LoadSessionFromSavedModel(
      options, 
      run, // NULL, // const TF_Buffer* run_options,
      argv[1], 
      tags, 
      1,
      graph, 
      meta, // NULL, //TF_Buffer* meta_graph_def, 
      status
   );

   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
   }

   // obtain input and output nodes
   TF_Output inputs[1];
   inputs[0].oper = TF_GraphOperationByName(graph, "in");
   inputs[0].index = 0;

   TF_Output outputs[1];
   outputs[0].oper = TF_GraphOperationByName(graph, "out");
   outputs[0].index = 0;

   /*
   // you can get dimensions from model like that:
   // (adjust dims to n for n>2)

   int n = TF_GraphGetTensorNumDims(graph, inputs[0], status);
   int64_t dims[2];
   TF_GraphGetTensorShape(graph, outputs[0], dims, 2, status);
   printf("%d: %d x %d\n", n, dims[0], dims[1]);
   */

   // allocate tensorts for input (1x784) and output (1x10)
   int64_t in_dims[2]  = { 1, 784 };
   int64_t out_dims[2] = { 1, 10 };

   TF_Tensor* input_values[1];
   TF_Tensor* output_values[1];

   input_values[0]  = TF_AllocateTensor(TF_FLOAT, in_dims, 2, 784*sizeof(float)); 
   output_values[0] = TF_AllocateTensor(TF_FLOAT, out_dims, 2, 10*sizeof(float)); 

   // write test vector to TF_TensorData(input_values[0])
   float* input_data = TF_TensorData(input_values[0]);
   for (int i=0; i<784; i++) 
   {  
      input_data[i] = 0.5;
   }

   // evaluate test tensor
   TF_SessionRun(
       tfs,
       // RunOptions
       run,
       // Input tensors
       inputs, 
       input_values, 
       1, 
       // Output tensors
       outputs, 
       output_values, 
       1, 
       // Target operations
       NULL, 
       0, 
       // RunMetadata
       NULL,
       // Output status
       status
   );
   
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
   }

   // print output/prediction
   float* output_data = TF_TensorData(output_values[0]);
   for (int i=0; i<10; i++) 
   {  
      printf("%f%s", output_data[i], (i==9) ? "\n" : ", ");
   }

   // terminate
   TF_CloseSession(tfs, status);
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
   }
   TF_DeleteSession(tfs, status);
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
   }
   TF_DeleteTensor(input_values[0]);
   TF_DeleteTensor(output_values[0]);
   TF_DeleteBuffer(meta);
   TF_DeleteBuffer(run);
   TF_DeleteSessionOptions(options);
   TF_DeleteGraph(graph);
   TF_DeleteStatus(status);

   return 0;
}

