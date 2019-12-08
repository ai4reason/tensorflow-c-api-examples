#include <stdio.h>
#include <tensorflow/c/c_api.h>
   
static int32_t ini_nodes[61] = {0,0,1,0,1,1,2,1,2,1,2,2,1,0,1,2,2,1,0,1,2,2,1,0,0,1,2,2,1,0,1,2,2,1,0,1,2,2,1,1,1,0,0,1,2,2,1,1,1,2,2,1,1,1,0,0,1,2,1,1,1};
static int32_t ini_symbols[18] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1};
static int32_t ini_clauses[15] = {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static int32_t node_inputs_1_lens[61] = {1,1,1,1,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1};
static int32_t node_inputs_1_symbols[42] = {0,1,8,2,9,10,11,12,12,3,13,12,3,14,12,3,4,9,12,3,15,12,3,16,12,11,8,3,5,9,8,8,17,17,10,8,7,6,9,11,8,9};
static int32_t node_inputs_1_nodes[84] = {-1,-1,-1,-1,0,1,-1,-1,3,0,1,-1,6,-1,8,6,10,11,11,-1,13,-1,15,16,16,-1,18,-1,20,21,21,-1,23,-1,24,1,26,27,27,-1,29,-1,31,32,32,-1,34,-1,36,37,36,-1,36,1,37,-1,41,36,42,37,44,45,45,44,0,1,49,50,50,-1,49,50,-1,-1,54,-1,1,55,57,-1,57,1,57,54};
static float node_inputs_1_sgn[42] = {1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1};
static int32_t node_inputs_2_lens[61] = {2,2,0,1,0,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,3,1,0,0,0,1,1,0,1,1,0,0,0,2,1,0,0,0,1,0,0,3,0,0,0};
static int32_t node_inputs_2_symbols[38] = {8,17,10,9,9,11,12,12,3,13,12,3,14,12,3,4,9,12,3,15,12,3,16,12,11,8,3,5,9,8,8,17,8,10,6,11,8,9};
static int32_t node_inputs_2_nodes[76] = {2,1,48,1,5,-1,56,55,4,0,7,-1,9,6,12,11,13,-1,14,-1,17,16,18,-1,19,-1,22,21,23,-1,24,-1,25,1,28,27,29,-1,30,-1,33,32,34,-1,35,-1,38,37,39,-1,40,1,41,-1,42,36,43,37,46,45,47,44,51,50,53,50,52,-1,55,-1,58,-1,59,1,60,54};
static float node_inputs_2_sgn[38] = {-1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,1};
static int32_t node_inputs_3_lens[61] = {1,5,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,2,0,0,0,0,0,0,1,1,0,0,0,0,2,0,0,0,1,1,0,0,0,0,0};
static int32_t node_inputs_3_symbols[21] = {9,8,8,8,9,17,12,12,12,12,12,12,5,12,9,8,8,17,8,9,9};
static int32_t node_inputs_3_nodes[42] = {4,3,2,0,40,36,59,57,25,24,48,0,9,8,12,10,17,15,22,20,28,26,33,31,42,41,38,36,43,42,47,45,46,44,51,49,53,49,60,57,56,1};
static float node_inputs_3_sgn[21] = {1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1};
static int32_t symbol_inputs_lens[18] = {1,1,1,6,1,1,1,1,6,5,2,3,7,1,1,1,1,2};
static int32_t symbol_inputs_nodes[126] = {0,-1,-1,1,-1,-1,3,-1,-1,13,11,-1,18,16,-1,23,21,-1,29,27,-1,34,32,-1,41,37,-1,24,23,-1,42,41,36,55,54,-1,54,-1,-1,2,0,1,40,36,1,46,44,45,47,45,44,59,57,1,53,49,50,56,1,55,4,3,0,25,24,1,43,42,37,60,57,54,5,1,-1,52,50,-1,7,6,-1,39,36,-1,58,57,-1,12,10,11,17,15,16,22,20,21,28,26,27,33,31,32,38,36,37,9,8,6,14,13,-1,19,18,-1,30,29,-1,35,34,-1,51,49,50,48,0,1};
static float symbol_inputs_sgn[42] = {1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,1};
static int32_t node_c_inputs_lens[61] = {0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,1};
static int32_t node_c_inputs_data[29] = {0,1,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,9,9,10,10,11,12,12,12,13,14,14,14};
static int32_t clause_inputs_lens[15] = {1,1,1,2,2,2,2,2,2,4,2,1,3,1,3};
static int32_t clause_inputs_data[29] = {2,4,5,7,9,12,14,17,19,22,25,28,30,33,35,38,39,40,43,46,47,48,51,52,53,56,58,59,60};
static int32_t prob_segments_lens[1] = {9};
static int32_t prob_segments_data[9] = {1,1,1,7,1,1,1,1,1};
static int32_t labels[8] = {1,1,1,1,1,1,1,1};



void load_vector_int32(int idx, TF_Graph* graph, TF_Output* inputs, TF_Tensor** input_values, int size, int32_t* data, char* name)
{
   inputs[idx].oper = TF_GraphOperationByName(graph, name);
   inputs[idx].index = 0;

   int64_t dims[1];
   dims[0] = size;
   input_values[idx] = TF_AllocateTensor(TF_INT32, dims, 1, size*sizeof(int32_t));
   int32_t* input_data = TF_TensorData(input_values[idx]);
   for (int i=0; i<size; i++) 
   {  
      input_data[i] = data[i];
   }
}

void load_vector_float(int idx, TF_Graph* graph, TF_Output* inputs, TF_Tensor** input_values, int size, float* data, char* name)
{
   inputs[idx].oper = TF_GraphOperationByName(graph, name);
   inputs[idx].index = 0;

   int64_t dims[1];
   dims[0] = size;
   input_values[idx] = TF_AllocateTensor(TF_FLOAT, dims, 1, size*sizeof(float));
   float* input_data = TF_TensorData(input_values[idx]);
   for (int i=0; i<size; i++) 
   {  
      input_data[i] = data[i];
   }
}

void load_matrix(int idx, TF_Graph* graph, TF_Output* inputs, TF_Tensor** input_values, int dimx, int dimy, int32_t* data, char* name)
{
   inputs[idx].oper = TF_GraphOperationByName(graph, name);
   inputs[idx].index = 0;

   int64_t dims[2];
   dims[0] = dimx;
   dims[1] = dimy;
   int64_t size = dimx * dimy;
   input_values[idx] = TF_AllocateTensor(TF_INT32, dims, 2, size*sizeof(int32_t));
   int32_t* input_data = TF_TensorData(input_values[idx]);
   for (int i=0; i<size; i++) 
   {  
      input_data[i] = data[i];
   }
}

void load_example_input(TF_Graph* graph, TF_Output* inputs, TF_Tensor** input_values)
{
   load_vector_int32(0, graph, inputs, input_values, 61, ini_nodes, "GraphPlaceholder/ini_nodes");
   load_vector_int32(1, graph, inputs, input_values, 18, ini_symbols, "GraphPlaceholder/ini_symbols");
   load_vector_int32(2, graph, inputs, input_values, 15, ini_clauses, "GraphPlaceholder/ini_clauses");
   load_vector_int32(3, graph, inputs, input_values, 61, node_inputs_1_lens, "GraphPlaceholder/GraphHyperEdgesA/segment_lens");
   load_vector_int32(4, graph, inputs, input_values, 42, node_inputs_1_symbols, "GraphPlaceholder/GraphHyperEdgesA/symbols");
   load_vector_float(5, graph, inputs, input_values, 42, node_inputs_1_sgn, "GraphPlaceholder/GraphHyperEdgesA/sgn");
   load_vector_int32(6, graph, inputs, input_values, 61, node_inputs_2_lens, "GraphPlaceholder/GraphHyperEdgesA_1/segment_lens");
   load_vector_int32(7, graph, inputs, input_values, 38, node_inputs_2_symbols, "GraphPlaceholder/GraphHyperEdgesA_1/symbols");
   load_vector_float(8, graph, inputs, input_values, 38, node_inputs_2_sgn, "GraphPlaceholder/GraphHyperEdgesA_1/sgn");
   load_vector_int32(9, graph, inputs, input_values, 61, node_inputs_3_lens, "GraphPlaceholder/GraphHyperEdgesA_2/segment_lens");
   load_vector_int32(10, graph, inputs, input_values, 21, node_inputs_3_symbols, "GraphPlaceholder/GraphHyperEdgesA_2/symbols");
   load_vector_float(11, graph, inputs, input_values, 21, node_inputs_3_sgn, "GraphPlaceholder/GraphHyperEdgesA_2/sgn");
   load_vector_int32(12, graph, inputs, input_values, 18, symbol_inputs_lens, "GraphPlaceholder/GraphHyperEdgesB/segment_lens");
   load_vector_float(13, graph, inputs, input_values, 42, symbol_inputs_sgn, "GraphPlaceholder/GraphHyperEdgesB/sgn");
   load_vector_int32(14, graph, inputs, input_values, 61, node_c_inputs_lens, "GraphPlaceholder/GraphEdges/segment_lens");
   load_vector_int32(15, graph, inputs, input_values, 29, node_c_inputs_data, "GraphPlaceholder/GraphEdges/data");
   load_vector_int32(16, graph, inputs, input_values, 15, clause_inputs_lens, "GraphPlaceholder/GraphEdges_1/segment_lens");
   load_vector_int32(17, graph, inputs, input_values, 29, clause_inputs_data, "GraphPlaceholder/GraphEdges_1/data");
   load_vector_int32(18, graph, inputs, input_values, 1, prob_segments_lens, "segment_lens");
   load_vector_int32(19, graph, inputs, input_values, 9, prob_segments_data, "segment_data");
   load_vector_int32(20, graph, inputs, input_values, 8, labels, "Placeholder");
   load_matrix(21, graph, inputs, input_values, 42, 2, node_inputs_1_nodes, "GraphPlaceholder/GraphHyperEdgesA/nodes");
   load_matrix(22, graph, inputs, input_values, 38, 2, node_inputs_2_nodes, "GraphPlaceholder/GraphHyperEdgesA_1/nodes");
   load_matrix(23, graph, inputs, input_values, 21, 2, node_inputs_3_nodes, "GraphPlaceholder/GraphHyperEdgesA_2/nodes");
   load_matrix(24, graph, inputs, input_values, 42, 3, symbol_inputs_nodes, "GraphPlaceholder/GraphHyperEdgesB/nodes");
}

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
      return -1;
   }

   // inputs
   TF_Output inputs[25];
   TF_Tensor* input_values[25];
   load_example_input(graph, inputs, input_values);

   // output
   TF_Output outputs[1];
   TF_Tensor* output_values[1];
   outputs[0].oper = TF_GraphOperationByName(graph, "Squeeze");
   outputs[0].index = 0;
   int64_t out_dims[1] = { 8 };
   output_values[0] = TF_AllocateTensor(TF_FLOAT, out_dims, 1, 8*sizeof(float)); 

   // evaluate test tensor
   TF_SessionRun(
       tfs,
       // RunOptions
       run,
       // Input tensors
       inputs, 
       input_values, 
       25, 
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
      return -1;
   }

   // print output/prediction
   float* output_data = TF_TensorData(output_values[0]);
   for (int i=0; i<8; i++) 
   {  
      printf("%f%s", output_data[i], ((i+1)%8==0) ? "\n" : ", ");
   }

   // terminate
   TF_CloseSession(tfs, status);
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
      return -1;
   }
   TF_DeleteSession(tfs, status);
   if (TF_GetCode(status) != TF_OK)
   {
      printf("Error: Tensorflow: %s\n", TF_Message(status));
      return -1;
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

