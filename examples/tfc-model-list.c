#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main(int argc, char** argv) 
{
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

   size_t pos = 0;
   TF_Operation* oper;
   while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL) 
   {
      printf("op: name=%s ins=%d outs=%d\n", TF_OperationName(oper),
            TF_OperationNumInputs(oper), TF_OperationNumOutputs(oper));
   }

   return 0;
}

