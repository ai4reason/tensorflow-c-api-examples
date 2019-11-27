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

   // default tags
   const char* tags[1] = {"serve"};

   // load session
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

   // list operations
   size_t pos = 0;
   TF_Operation* oper;
   while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL) 
   {
      printf("op: name=%s ins=%d outs=%d\n", TF_OperationName(oper),
            TF_OperationNumInputs(oper), TF_OperationNumOutputs(oper));
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
   TF_DeleteBuffer(meta);
   TF_DeleteBuffer(run);
   TF_DeleteSessionOptions(options);
   TF_DeleteGraph(graph);
   TF_DeleteStatus(status);

   return 0;
}

